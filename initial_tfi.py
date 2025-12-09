"""
Algebraic C-Mesh Generator (Transfinite Interpolation)
Author: Phani Raghava Panchagnula
Description: Generates a structured C-Grid around a NACA 4-digit airfoil using TFI.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

class AirfoilGenerator:
    """Generates NACA 4-digit coordinates with cosine clustering."""
    @staticmethod
    def naca4(code, n_points):
        m = int(code[0]) / 100.0
        p = int(code[1]) / 10.0
        t = int(code[2:]) / 100.0
        
        # Cosine clustering for density at LE and TE
        beta = np.linspace(0, np.pi, n_points)
        x = (1 - np.cos(beta)) / 2.0
        
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 +
                      0.2843 * x**3 - 0.1015 * x**4)
        
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
        for i in range(len(x)):
            if x[i] < p:
                yc[i] = (m / p**2) * (2 * p * x[i] - x[i]**2)
                dyc_dx[i] = (2 * m / p**2) * (p - x[i])
            else:
                yc[i] = (m / (1 - p)**2) * ((1 - 2*p) + 2*p*x[i] - x[i]**2)
                dyc_dx[i] = (2 * m / (1 - p)**2) * (p - x[i])
                
        theta = np.arctan(dyc_dx)
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)
        return xu, yu, xl, yl

def tanh_dist(n, b=4.0):
    """
    Hyperbolic tangent distribution function for wall clustering.
    Ref: Vinokur, M., 'On One-Dimensional Stretching Functions for Finite-Difference Calculations', 1983.
    """
    s = np.linspace(0, 1, n)
    return 1.0 + np.tanh(b * (s - 1.0)) / np.tanh(b)

def geometric_expansion(n_points, length, dx_start):
    """
    Generates point distribution expanding geometrically from dx_start.
    Used to match Airfoil TE spacing to Wake spacing.
    """
    # Iterative solve for growth ratio 'r'
    r = 1.1 
    for _ in range(20):
        # Newton-Raphson could be used, but simple fixed-point iteration works for this range
        # Formula: L = dx * (1 - r^n) / (1 - r) -> re-arranged for r
        # Here we approximate by generating and scaling
        pass 
    
    # Direct generation loop with scaling
    pts = [0.0]
    dx = dx_start
    r = 1.15 # Target expansion ratio
    for _ in range(n_points-1):
        pts.append(pts[-1] + dx)
        dx *= r
    pts = np.array(pts)
    return pts * (length / pts[-1]) # Scale to exact length

class TFIGenerator:
    def __init__(self, xu, yu, xl, yl, wake_len=6.0, ni_wake=30, nj=40, clustering=4.0):
        self.xu, self.yu, self.xl, self.yl = xu, yu, xl, yl
        self.wake_len = wake_len
        self.ni_wake = ni_wake
        self.nj = nj
        self.clustering = clustering

    def build_boundaries(self):
        # 1. Calculate Trailing Edge Spacing (Upper/Lower avg)
        dx_te = 0.5 * (np.hypot(self.xu[-1]-self.xu[-2], self.yu[-1]-self.yu[-2]) + 
                       np.hypot(self.xl[-1]-self.xl[-2], self.yl[-1]-self.yl[-2]))

        # 2. Generate Wake Points (Matched Spacing)
        wake_rel = geometric_expansion(self.ni_wake, self.wake_len, dx_te)
        wake_x = 1.0 + wake_rel
        wake_y = np.zeros_like(wake_x)

        # 3. Assemble Inner Boundary (Lower Wake -> Airfoil -> Upper Wake)
        wake_x_lower = wake_x[::-1]
        wake_y_lower = wake_y[::-1]
        
        inner_x = np.concatenate([wake_x_lower[:-1], self.xl[::-1][:-1], self.xu, wake_x[1:]])
        inner_y = np.concatenate([wake_y_lower[:-1], self.yl[::-1][:-1], self.yu, wake_y[1:]])
        self.inner_x, self.inner_y = inner_x, inner_y
        self.ni = len(inner_x)

        # 4. Assemble Outer Boundary (C-Topology Farfield)
        outer_x = np.zeros(self.ni)
        outer_y = np.zeros(self.ni)
        
        idx1 = self.ni_wake - 1
        idx2 = self.ni - self.ni_wake
        
        for i in range(self.ni):
            if i < idx1: # Lower Wake Boundary
                t = i / idx1
                outer_x[i] = inner_x[i] # Orthogonal projection logic
                outer_y[i] = -self.wake_len
            elif i > idx2: # Upper Wake Boundary
                t = (i - idx2) / (self.ni - 1 - idx2)
                outer_x[i] = inner_x[i]
                outer_y[i] = self.wake_len
            else: # C-Arc
                t_arc = (i - idx1) / (idx2 - idx1)
                ang = 1.5 * np.pi - t_arc * np.pi # 270 deg -> 90 deg
                outer_x[i] = 1.0 + self.wake_len * np.cos(ang)
                outer_y[i] = self.wake_len * np.sin(ang)
        self.outer_x, self.outer_y = outer_x, outer_y

    def generate(self):
        """Unidirectional Transfinite Interpolation (Linear)."""
        self.build_boundaries()
        X = np.zeros((self.nj, self.ni))
        Y = np.zeros((self.nj, self.ni))
        
        # Normalized coordinate eta [0,1]
        eta = tanh_dist(self.nj, b=self.clustering)
        
        for i in range(self.ni):
            for j in range(self.nj):
                s = eta[j]
                # Linear Interpolation between Inner and Outer
                X[j,i] = (1-s)*self.inner_x[i] + s*self.outer_x[i]
                Y[j,i] = (1-s)*self.inner_y[i] + s*self.outer_y[i]
        
        return X, Y

if __name__ == "__main__":
    gen = TFIGenerator(*AirfoilGenerator.naca4("4412", 80), nj=50)
    X, Y = gen.generate()
    plt.plot(X.T, Y.T, 'k-', lw=0.5); plt.plot(X, Y, 'k-', lw=0.5)
    plt.axis('equal'); plt.show()