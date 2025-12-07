import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- 1. Math Utilities ---

class GridUtils:
    @staticmethod
    def tanh_distribution(n, clustering_strength=2.5):
        """Clustering at the end (1.0). Used for Wall Normal direction."""
        x = np.linspace(0, 1, n)
        return 1.0 + np.tanh(clustering_strength * (x - 1.0)) / np.tanh(clustering_strength)

    @staticmethod
    def geometric_expansion(n_points, length, dx_start):
        """
        Generates points 0..length with exact starting spacing dx_start.
        Used to match Wake cells to Airfoil cells.
        """
        # Solves for growth rate 'r' such that sum of geometric series equals length
        # This is an approximation for valid r > 1.0
        if n_points < 2: return np.array([0.0, length])
        
        # Iterative solver for r: dx * (1-r^n)/(1-r) = L
        r = 1.1 # Initial guess
        for _ in range(10):
            r = (length / dx_start * (r - 1) + 1)**(1/(n_points-1))
        
        # Generate points
        points = [0.0]
        curr = 0.0
        dx = dx_start
        for _ in range(n_points - 1):
            curr += dx
            points.append(curr)
            dx *= r
        
        # Normalize to ensure exact length match despite precision errors
        points = np.array(points)
        points = points * (length / points[-1])
        return points

    @staticmethod
    def bezier_curve(p0, p1, p2, t_values):
        """Quadratic Bezier."""
        curve = np.zeros((len(t_values), 2))
        for i, t in enumerate(t_values):
            curve[i] = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
        return curve[:, 0], curve[:, 1]

# --- 2. Geometry Engine ---

class AirfoilGenerator:
    @staticmethod
    def naca4(number, n_points):
        m = int(number[0]) / 100.0
        p = int(number[1]) / 10.0
        t = int(number[2:]) / 100.0
        
        # Cosine clustering for Airfoil
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
        
        te_slope = (2 * m / (1 - p)**2) * (p - 1.0) if p < 1.0 else 0
        te_angle = np.arctan(te_slope)

        return xu, yu, xl, yl, te_angle

# --- 3. Mesher ---

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

    def generate_boundary_points(self, wake_type='curved', clustering_strength=3.0):
        
        # --- 1. Wake Distribution Matching ---
        # Calculate dx at the Trailing Edge of the airfoil (Upper surface end)
        dx_te_airfoil = np.sqrt((self.xu[-1] - self.xu[-2])**2 + (self.yu[-1] - self.yu[-2])**2)
        
        # Generate Wake Spacing that matches this dx_te
        # This fixes the "Mismatch"
        wake_dist_linear = GridUtils.geometric_expansion(self.ni_wake, self.wake_len, dx_te_airfoil)
        
        # Normalize to 0..1 for Bezier interpolation
        t_wake = wake_dist_linear / self.wake_len 

        # --- 2. Wake Geometry ---
        if wake_type == 'curved':
            p0 = np.array([1.0, 0.0])
            p1 = np.array([1.0 + self.wake_len * 0.3, 0.0 + (self.wake_len * 0.3) * np.tan(self.te_angle)])
            p2 = np.array([1.0 + self.wake_len, 0.0])
            p2[1] = p1[1] * 0.5 
            
            wx, wy = GridUtils.bezier_curve(p0, p1, p2, t_wake)
            wake_x_lower, wake_y_lower = wx[::-1], wy[::-1]
            wake_x_upper, wake_y_upper = wx, wy
        else:
            wake_x = 1.0 + wake_dist_linear
            wake_y = np.zeros_like(wake_x)
            wake_x_lower, wake_y_lower = wake_x[::-1], wake_y[::-1]
            wake_x_upper, wake_y_upper = wake_x, wake_y

        # --- 3. Combine Inner Boundary ---
        self.inner_x = np.concatenate([wake_x_lower[:-1], self.xl[::-1][:-1], self.xu, wake_x_upper[1:]])
        self.inner_y = np.concatenate([wake_y_lower[:-1], self.yl[::-1][:-1], self.yu, wake_y_upper[1:]])
        self.ni = len(self.inner_x)

        # --- 4. Outer Boundary ---
        self.outer_x = np.zeros(self.ni)
        self.outer_y = np.zeros(self.ni)
        
        idx_wake_end_lower = self.ni_wake - 1
        idx_wake_start_upper = self.ni - self.ni_wake

        for i in range(self.ni):
            if i < idx_wake_end_lower: 
                t = i / idx_wake_end_lower
                self.outer_x[i] = (1.0 + self.wake_len)*(1-t) + 1.0*t
                self.outer_y[i] = -self.radius
            elif i > idx_wake_start_upper: 
                t = (i - idx_wake_start_upper) / (self.ni - 1 - idx_wake_start_upper)
                self.outer_x[i] = 1.0*(1-t) + (1.0 + self.wake_len)*t
                self.outer_y[i] = self.radius
            else: 
                t_arc = (i - idx_wake_end_lower) / (idx_wake_start_upper - idx_wake_end_lower)
                angle = (3*np.pi/2) - t_arc * np.pi 
                self.outer_x[i] = 1.0 + self.radius * np.cos(angle)
                self.outer_y[i] = self.radius * np.sin(angle)
        
        self.clustering_strength = clustering_strength

    def generate_algebraic(self):
        self.X = np.zeros((self.nj, self.ni))
        self.Y = np.zeros((self.nj, self.ni))
        dist = GridUtils.tanh_distribution(self.nj, self.clustering_strength)

        for i in range(self.ni):
            for j in range(self.nj):
                s = dist[j]
                self.X[j, i] = self.inner_x[i] + s * (self.outer_x[i] - self.inner_x[i])
                self.Y[j, i] = self.inner_y[i] + s * (self.outer_y[i] - self.inner_y[i])

    def smooth_elliptic(self, iterations=50, freeze_layers=10):
        print(f"  > Smoothing ({iterations} iters)...")
        X_new, Y_new = self.X.copy(), self.Y.copy()
        
        for it in range(iterations):
            for j in range(freeze_layers, self.nj - 1):
                for i in range(1, self.ni - 1):
                    x_xi = 0.5 * (self.X[j, i+1] - self.X[j, i-1])
                    x_eta = 0.5 * (self.X[j+1, i] - self.X[j-1, i])
                    y_xi = 0.5 * (self.Y[j, i+1] - self.Y[j, i-1])
                    y_eta = 0.5 * (self.Y[j+1, i] - self.Y[j-1, i])
                    
                    alpha = x_eta**2 + y_eta**2
                    beta  = x_xi * x_eta + y_xi * y_eta
                    gamma = x_xi**2 + y_xi**2
                    
                    x_next = (alpha * (self.X[j, i+1] + self.X[j, i-1]) + 
                              gamma * (self.X[j+1, i] + self.X[j-1, i]) - 
                              2 * beta * 0.25 * (self.X[j+1, i+1] - self.X[j-1, i+1] - self.X[j+1, i-1] + self.X[j-1, i-1])) / (2 * (alpha + gamma))
                    y_next = (alpha * (self.Y[j, i+1] + self.Y[j, i-1]) + 
                              gamma * (self.Y[j+1, i] + self.Y[j-1, i]) - 
                              2 * beta * 0.25 * (self.Y[j+1, i+1] - self.Y[j-1, i+1] - self.Y[j+1, i-1] + self.Y[j-1, i-1])) / (2 * (alpha + gamma))
                    
                    omega = 0.6
                    X_new[j, i] = (1-omega)*self.X[j, i] + omega*x_next
                    Y_new[j, i] = (1-omega)*self.Y[j, i] + omega*y_next
            self.X, self.Y = X_new.copy(), Y_new.copy()

# --- 4. Plotting ---

def plot_final(mesher, freeze_layers):
    X, Y = mesher.X, mesher.Y
    nj, ni = X.shape
    
    fig = plt.figure(figsize=(16, 7))
    
    # -- View 1: Global Mesh --
    ax1 = fig.add_subplot(1, 2, 1)
    
    # Decimate lines for cleaner global view
    step_i = max(1, int(ni/60))
    step_j = max(1, int(nj/40))
    
    for j in range(0, nj, step_j): ax1.plot(X[j, :], Y[j, :], 'k-', lw=0.3, alpha=0.4)
    for i in range(0, ni, step_i): ax1.plot(X[:, i], Y[:, i], 'k-', lw=0.3, alpha=0.4)
    
    # Draw Boundaries
    ax1.plot(X[-1, :], Y[-1, :], 'b-', lw=1.5, label='Farfield')
    ax1.plot(X[0, :], Y[0, :], 'r-', lw=1.5, label='Inner Boundary')
    
    # FIX 1: Set Limits explicitly so we ALWAYS see the blue farfield line
    margin = 1.0
    ax1.set_xlim(-mesher.radius - margin, mesher.radius + mesher.wake_len + margin)
    ax1.set_ylim(-mesher.radius - margin, mesher.radius + margin)
    
    ax1.set_title("Global Mesh (Farfield Visible)")
    ax1.set_aspect('equal')
    ax1.legend()

    # -- View 2: Trailing Edge Match --
    ax2 = fig.add_subplot(1, 2, 2)
    
    zoom_center = mesher.ni_wake - 1 # This is the TE index
    start = zoom_center - 10
    end = zoom_center + 15
    layers = freeze_layers + 5
    
    # Plot Prism Layers
    for j in range(layers):
        c = 'green' if j < freeze_layers else 'black'
        ax2.plot(X[j, start:end], Y[j, start:end], color=c, lw=0.8)
    
    # Plot Vertical Connectors to show alignment
    for i in range(start, end):
        ax2.plot(X[:layers, i], Y[:layers, i], 'k-', lw=0.4)

    # Highlight Surface vs Wake
    ax2.plot(X[0, start:zoom_center+1], Y[0, start:zoom_center+1], 'r-', lw=3, label='Airfoil')
    ax2.plot(X[0, zoom_center:], Y[0, zoom_center:], 'b--', lw=2, label='Wake Start')

    ax2.set_title("Trailing Edge Match\n(Note: Smooth cell size transition)")
    ax2.set_aspect('equal')
    # Auto-zoom on the TE region
    ax2.set_xlim(0.9, 1.2)
    ax2.set_ylim(-0.1, 0.1)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    naca_code = "4412"
    n_points = 100        # High resolution on airfoil
    wake_len = 8.0
    radius = 6.0
    
    ni_wake = 30
    nj = 60
    
    geom_data = AirfoilGenerator.naca4(naca_code, n_points)
    mesher = StructuredMesher(geom_data, radius, wake_len, ni_wake, nj)
    
    print("Generating Boundary with Matched Wake Spacing...")
    mesher.generate_boundary_points(wake_type='curved', clustering_strength=3.5)
    
    mesher.generate_algebraic()
    mesher.smooth_elliptic(iterations=30, freeze_layers=15)
    
    plot_final(mesher, freeze_layers=15)

if __name__ == "__main__":
    main()