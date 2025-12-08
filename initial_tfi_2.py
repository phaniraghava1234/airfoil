"""
Algebraic C-Mesh Generator (Transfinite Interpolation)
Contains the save_xyz function and the base TFI logic.

References:
1. Abbott, I. H., & Von Doenhoff, A. E. (1959). Theory of wing sections. Dover Publications.
2. Thompson, J. F., Warsi, Z. U., & Mastin, C. W. (1985). Numerical grid generation: foundations and applications.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

class AirfoilGenerator:
    """
    Generates NACA 4-digit coordinates with cosine clustering.
    
    Mathematical Basis:
    - Thickness distribution derived from Abbott & Von Doenhoff, Eq. 6.3/6.4.
    - Camber line equations for 4-digit series.
    """
    @staticmethod
    def naca4(code, n_points):
        # Parsing NACA 4-digit code (e.g., '4412')
        m = int(code[0]) / 100.0  # Max camber
        p = int(code[1]) / 10.0   # Location of max camber
        t = int(code[2:]) / 100.0 # Max thickness
        
        # Cosine clustering for higher resolution at LE/TE
        beta = np.linspace(0, np.pi, n_points)
        x = (1 - np.cos(beta)) / 2.0
        
        # Thickness distribution (Abbott & Von Doenhoff)
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 +
                      0.2843 * x**3 - 0.1015 * x**4)
        
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
        
        # Camber line calculations
        for i in range(len(x)):
            if x[i] < p:
                yc[i] = (m / p**2) * (2 * p * x[i] - x[i]**2)
                dyc_dx[i] = (2 * m / p**2) * (p - x[i])
            else:
                yc[i] = (m / (1 - p)**2) * ((1 - 2*p) + 2*p*x[i] - x[i]**2)
                dyc_dx[i] = (2 * m / (1 - p)**2) * (p - x[i])
                
        theta = np.arctan(dyc_dx)
        
        # Apply thickness perpendicular to camber line
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)
        return xu, yu, xl, yl

def tanh_dist(n, b=4.0):
    """
    Hyperbolic tangent stretching function for boundary layer clustering.
    Ref: Vinokur, M. (1983). 'On one-dimensional stretching functions for finite-difference calculations'.
    """
    s = np.linspace(0, 1, n)
    return 1.0 + np.tanh(b * (s - 1.0)) / np.tanh(b)

def geometric_expansion(n_points, length, dx_start):
    """Standard geometric series expansion for wake distribution."""
    pts = [0.0]
    dx = dx_start
    r = 1.15 # Growth rate
    for _ in range(n_points-1):
        pts.append(pts[-1] + dx)
        dx *= r
    pts = np.array(pts)
    return pts * (length / pts[-1])

def save_xyz(X, Y, filename):
    """Exports mesh to XYZ/CSV format."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['i', 'j', 'x', 'y'])
        nj, ni = X.shape
        for j in range(nj):
            for i in range(ni):
                writer.writerow([i, j, X[j,i], Y[j,i]])
    print(f"Saved {filename}")

class TFIGenerator:
    """
    Implements Transfinite Interpolation (TFI) for C-Mesh topology.
    This serves as the algebraic initial guess for the elliptic smoother.
    """
    def __init__(self, xu, yu, xl, yl, wake_len=6.0, ni_wake=30, nj=40, clustering=4.0):
        self.xu, self.yu, self.xl, self.yl = xu, yu, xl, yl
        self.wake_len = wake_len
        self.ni_wake = ni_wake
        self.nj = nj
        self.clustering = clustering

    def generate(self):
        # 1. Trailing Edge Spacing
        dx_te = 0.5 * (np.hypot(self.xu[-1]-self.xu[-2], self.yu[-1]-self.yu[-2]) + 
                       np.hypot(self.xl[-1]-self.xl[-2], self.yl[-1]-self.yl[-2]))

        # 2. Wake Points generation
        wake_rel = geometric_expansion(self.ni_wake, self.wake_len, dx_te)
        wake_x = 1.0 + wake_rel
        wake_y = np.zeros_like(wake_x)

        wake_x_lower = wake_x[::-1]
        wake_y_lower = wake_y[::-1]
        
        # Assemble inner boundary (Wake -> Lower Surface -> Upper Surface -> Wake)
        inner_x = np.concatenate([wake_x_lower[:-1], self.xl[::-1][:-1], self.xu, wake_x[1:]])
        inner_y = np.concatenate([wake_y_lower[:-1], self.yl[::-1][:-1], self.yu, wake_y[1:]])
        
        self.ni = len(inner_x)

        # 3. Outer Boundary Generation (C-Topology)
        outer_x = np.zeros(self.ni)
        outer_y = np.zeros(self.ni)
        
        idx1 = self.ni_wake - 1
        idx2 = self.ni - self.ni_wake
        
        for i in range(self.ni):
            if i < idx1:
                outer_x[i] = inner_x[i]
                outer_y[i] = -self.wake_len
            elif i > idx2:
                outer_x[i] = inner_x[i]
                outer_y[i] = self.wake_len
            else:
                # Circular arc for the C-mesh farfield
                t_arc = (i - idx1) / (idx2 - idx1)
                ang = 1.5 * np.pi - t_arc * np.pi
                outer_x[i] = 1.0 + self.wake_len * np.cos(ang)
                outer_y[i] = self.wake_len * np.sin(ang)

        # 4. Transfinite Interpolation (Algebraic Grid Generation)
        # Ref: Thompson et al., Eq. (Linear TFI in 2D)
        self.X = np.zeros((self.nj, self.ni))
        self.Y = np.zeros((self.nj, self.ni))
        eta = tanh_dist(self.nj, b=self.clustering)
        
        for i in range(self.ni):
            for j in range(self.nj):
                s = eta[j]
                # Simple linear interpolation between inner and outer boundaries
                self.X[j,i] = (1-s)*inner_x[i] + s*outer_x[i]
                self.Y[j,i] = (1-s)*inner_y[i] + s*outer_y[i]
        
        return self.X, self.Y

if __name__ == "__main__":
    gen = TFIGenerator(*AirfoilGenerator.naca4("4412", 80), nj=50)
    X, Y = gen.generate()
    save_xyz(X, Y, "tfi_test.csv")
    plt.plot(X.T, Y.T, 'k-', lw=0.5); plt.plot(X, Y, 'k-', lw=0.5)
    plt.axis('equal'); plt.show()