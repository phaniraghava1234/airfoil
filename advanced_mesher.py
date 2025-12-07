import numpy as np
import matplotlib.pyplot as plt
import sys

# --- 1. Math & Grid Utilities ---

class GridUtils:
    @staticmethod
    def tanh_distribution(n, clustering_strength=2.5):
        """
        One-sided hyperbolic tangent distribution.
        clustering_strength: 
            2.0 = Moderate clustering
            4.0 = Strong clustering (tight prism layer, coarse farfield)
        """
        x = np.linspace(0, 1, n)
        return 1.0 + np.tanh(clustering_strength * (x - 1.0)) / np.tanh(clustering_strength)

    @staticmethod
    def bezier_curve(p0, p1, p2, n_points):
        """Quadratic Bezier for smooth wake curvature."""
        t = np.linspace(0, 1, n_points)
        curve = np.zeros((n_points, 2))
        for i in range(n_points):
            curve[i] = (1-t[i])**2 * p0 + 2*(1-t[i])*t[i] * p1 + t[i]**2 * p2
        return curve[:, 0], curve[:, 1]

# --- 2. Geometry Engine ---

class AirfoilGenerator:
    @staticmethod
    def naca4(number, n_points):
        """
        Generates NACA 4-digit airfoil.
        Returns coordinates AND the Trailing Edge (TE) Bisector angle.
        """
        m = int(number[0]) / 100.0
        p = int(number[1]) / 10.0
        t = int(number[2:]) / 100.0
        
        # Use cosine clustering for points (dense at LE and TE)
        beta = np.linspace(0, np.pi, n_points)
        x = (1 - np.cos(beta)) / 2.0
        
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
        
        for i in range(len(x)):
            if x[i] < p:
                yc[i] = (m / p**2) * (2 * p * x[i] - x[i]**2)
                dyc_dx[i] = (2 * m / p**2) * (p - x[i])
            else:
                yc[i] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x[i] - x[i]**2)
                dyc_dx[i] = (2 * m / (1 - p)**2) * (p - x[i])
        
        theta = np.arctan(dyc_dx)
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)
        
        # Calculate TE exit angle (approximate via camber line slope at x=1)
        # dyc_dx at x=1 is the slope of the camber line at the tail
        te_slope = (2 * m / (1 - p)**2) * (p - 1.0) if p < 1.0 else 0
        te_angle = np.arctan(te_slope)

        return xu, yu, xl, yl, te_angle

# --- 3. The Advanced Mesher ---

class StructuredMesher:
    def __init__(self, airfoil_data, radius, wake_len, ni_wake, nj):
        self.xu, self.yu, self.xl, self.yl, self.te_angle = airfoil_data
        self.radius = radius
        self.wake_len = wake_len
        self.ni_wake = ni_wake
        self.nj = nj
        self.X = None
        self.Y = None
        self.ni = 0 

    def generate_boundary_points(self, wake_type='straight', clustering_strength=3.0):
        """
        Generates the Inner (Airfoil+Wake) and Outer (Farfield) boundaries.
        wake_type: 'straight' or 'curved' (follows airfoil camber)
        """
        
        # --- A. Generate Wake Lines ---
        if wake_type == 'curved':
            # Control points for Bezier Wake
            # P0: TE
            # P1: Control point projected out at TE angle
            # P2: End point (straight downstream)
            p0 = np.array([1.0, 0.0])
            p1 = np.array([1.0 + self.wake_len * 0.3, 0.0 + (self.wake_len * 0.3) * np.tan(self.te_angle)])
            p2 = np.array([1.0 + self.wake_len, 0.0]) # Force return to y=0 far downstream? Or keep offset?
            # Let's keep it somewhat aligned with flow, maybe slightly offset if highly cambered
            p2[1] = p1[1] * 0.5 # Decay to half height
            
            wx, wy = GridUtils.bezier_curve(p0, p1, p2, self.ni_wake)
            
            # Split into Upper and Lower (coincident for a sharp TE wake)
            wake_x_lower = wx[::-1] # Reverse for lower loop
            wake_y_lower = wy[::-1]
            wake_x_upper = wx
            wake_y_upper = wy
            
        else: # Straight
            wake_x = np.linspace(1.0, 1.0 + self.wake_len, self.ni_wake)
            wake_y = np.zeros_like(wake_x)
            wake_x_lower = wake_x[::-1]
            wake_y_lower = wake_y[::-1]
            wake_x_upper = wake_x
            wake_y_upper = wake_y

        # --- B. Combine into Inner Boundary ---
        # Note: xl is TE->LE, xu is LE->TE.
        self.inner_x = np.concatenate([wake_x_lower[:-1], self.xl[::-1][:-1], self.xu, wake_x_upper[1:]])
        self.inner_y = np.concatenate([wake_y_lower[:-1], self.yl[::-1][:-1], self.yu, wake_y_upper[1:]])
        self.ni = len(self.inner_x)

        # --- C. Outer Boundary (Farfield) ---
        self.outer_x = np.zeros(self.ni)
        self.outer_y = np.zeros(self.ni)
        
        idx_wake_end_lower = self.ni_wake - 1
        idx_wake_start_upper = self.ni - self.ni_wake

        for i in range(self.ni):
            if i < idx_wake_end_lower: # Lower Wake
                t = i / idx_wake_end_lower
                # Interpolate from Farfield Bottom to C-Circle Start
                self.outer_x[i] = (1.0 + self.wake_len)*(1-t) + 1.0*t
                self.outer_y[i] = -self.radius
            elif i > idx_wake_start_upper: # Upper Wake
                t = (i - idx_wake_start_upper) / (self.ni - 1 - idx_wake_start_upper)
                self.outer_x[i] = 1.0*(1-t) + (1.0 + self.wake_len)*t
                self.outer_y[i] = self.radius
            else: # C-Shape around Airfoil
                t_arc = (i - idx_wake_end_lower) / (idx_wake_start_upper - idx_wake_end_lower)
                # Map 270 -> 90 degrees
                angle = (3*np.pi/2) - t_arc * np.pi 
                self.outer_x[i] = 1.0 + self.radius * np.cos(angle)
                self.outer_y[i] = self.radius * np.sin(angle)

        # Save clustering param for TFI step
        self.clustering_strength = clustering_strength

    def generate_algebraic(self):
        """Standard TFI."""
        self.X = np.zeros((self.nj, self.ni))
        self.Y = np.zeros((self.nj, self.ni))
        
        # Use tunable clustering here
        dist = GridUtils.tanh_distribution(self.nj, self.clustering_strength)

        for i in range(self.ni):
            for j in range(self.nj):
                s = dist[j]
                self.X[j, i] = self.inner_x[i] + s * (self.outer_x[i] - self.inner_x[i])
                self.Y[j, i] = self.inner_y[i] + s * (self.outer_y[i] - self.inner_y[i])

    def smooth_elliptic(self, iterations=50, freeze_layers=10):
        """Winslow Smoother with Frozen Prism Layers."""
        print(f"  > Running Elliptic Smoother ({iterations} iters)...")
        X_new = self.X.copy()
        Y_new = self.Y.copy()
        
        for it in range(iterations):
            max_diff = 0.0
            # Skip boundaries (j=0, j=max) and Prism Layers (j < freeze)
            for j in range(freeze_layers, self.nj - 1):
                for i in range(1, self.ni - 1):
                    # Central Differences
                    x_xi = 0.5 * (self.X[j, i+1] - self.X[j, i-1])
                    x_eta = 0.5 * (self.X[j+1, i] - self.X[j-1, i])
                    y_xi = 0.5 * (self.Y[j, i+1] - self.Y[j, i-1])
                    y_eta = 0.5 * (self.Y[j+1, i] - self.Y[j-1, i])
                    
                    # Coefficients
                    alpha = x_eta**2 + y_eta**2
                    beta  = x_xi * x_eta + y_xi * y_eta
                    gamma = x_xi**2 + y_xi**2
                    
                    # Solver Step
                    x_next = (alpha * (self.X[j, i+1] + self.X[j, i-1]) + 
                              gamma * (self.X[j+1, i] + self.X[j-1, i]) - 
                              2 * beta * 0.25 * (self.X[j+1, i+1] - self.X[j-1, i+1] - self.X[j+1, i-1] + self.X[j-1, i-1])) / (2 * (alpha + gamma))
                    y_next = (alpha * (self.Y[j, i+1] + self.Y[j, i-1]) + 
                              gamma * (self.Y[j+1, i] + self.Y[j-1, i]) - 
                              2 * beta * 0.25 * (self.Y[j+1, i+1] - self.Y[j-1, i+1] - self.Y[j+1, i-1] + self.Y[j-1, i-1])) / (2 * (alpha + gamma))
                    
                    # Under-relaxation
                    omega = 0.6
                    X_new[j, i] = (1-omega)*self.X[j, i] + omega*x_next
                    Y_new[j, i] = (1-omega)*self.Y[j, i] + omega*y_next
                    
            self.X = X_new.copy()
            self.Y = Y_new.copy()

# --- 4. Plotting & Execution ---

def plot_final(mesher, freeze_layers):
    X, Y = mesher.X, mesher.Y
    nj, ni = X.shape
    
    fig = plt.figure(figsize=(15, 7))
    
    # -- View 1: Global Mesh --
    ax1 = fig.add_subplot(1, 2, 1)
    # Skip lines for visibility
    step_i = max(1, int(ni/50))
    step_j = max(1, int(nj/30))
    
    for j in range(0, nj, step_j): ax1.plot(X[j, :], Y[j, :], 'k-', lw=0.3, alpha=0.5)
    for i in range(0, ni, step_i): ax1.plot(X[:, i], Y[:, i], 'k-', lw=0.3, alpha=0.5)
    
    # Boundaries
    ax1.plot(X[-1, :], Y[-1, :], 'b-', lw=1.5, label='Farfield')
    ax1.plot(X[0, :], Y[0, :], 'k--', lw=0.8, label='Wake Line') # Plot full wake
    
    # Highlight Airfoil explicitly
    start_af = mesher.ni_wake - 1
    end_af = ni - mesher.ni_wake + 1
    ax1.plot(X[0, start_af:end_af], Y[0, start_af:end_af], 'r-', lw=2, label='NACA 4412')

    ax1.set_title("Global C-Mesh (NACA 4412)")
    ax1.set_aspect('equal')
    ax1.legend()

    # -- View 2: Trailing Edge Detail (Showing Prism Layer) --
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Zoom in on TE to see wake curvature and prism layers
    zoom_start = mesher.ni_wake - 15
    zoom_end = mesher.ni_wake + 15
    zoom_layers = freeze_layers + 10
    
    # Prism Layers (Green)
    for j in range(zoom_layers):
        c = 'green' if j < freeze_layers else 'black'
        lw = 1.0 if j < freeze_layers else 0.5
        ax2.plot(X[j, zoom_start:zoom_end], Y[j, zoom_start:zoom_end], color=c, lw=lw)
        
    for i in range(zoom_start, zoom_end):
        ax2.plot(X[:zoom_layers, i], Y[:zoom_layers, i], 'k-', lw=0.3)
        
    ax2.plot(X[0, start_af:end_af], Y[0, start_af:end_af], 'r-', lw=3)
    
    ax2.set_title(f"Trailing Edge Zoom\n(Green = {freeze_layers} Frozen Prism Layers)")
    ax2.set_aspect('equal')
    ax2.set_xlim(0.8, 1.4)
    ax2.set_ylim(-0.2, 0.2)
    
    plt.tight_layout()
    plt.show()

def main():
    # --- Configuration ---
    naca_code = "4412"    # Highly cambered
    n_points = 65         # Points on airfoil
    wake_len = 6.0
    radius = 6.0
    
    ni_wake = 25          # Points along wake
    nj = 50               # Normal layers (Wall -> Farfield)
    
    # Refinement Params
    clustering_val = 1.0 # HIGHER = Finer at wall, Coarser at farfield (Refines "away" from farfield)
    wake_mode = 'straight'  # 'straight' or 'curved'
    frozen_layers = 12    # How many layers to protect from smoothing

    print(f"Generating Mesh for NACA {naca_code}...")
    
    # 1. Geometry
    # Unpack 5 values now (includes TE angle)
    geom_data = AirfoilGenerator.naca4(naca_code, n_points)
    
    # 2. Initialize
    mesher = StructuredMesher(geom_data, radius, wake_len, ni_wake, nj)
    
    # 3. Generate Boundaries (Wake shape decided here)
    mesher.generate_boundary_points(wake_type=wake_mode, clustering_strength=clustering_val)
    
    # 4. Algebraic Guess
    mesher.generate_algebraic()
    
    # 5. Elliptic Smoothing
    mesher.smooth_elliptic(iterations=40, freeze_layers=frozen_layers)
    
    # 6. Visualize
    plot_final(mesher, frozen_layers)

if __name__ == "__main__":
    main()