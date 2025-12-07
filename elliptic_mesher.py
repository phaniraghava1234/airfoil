import numpy as np
import matplotlib.pyplot as plt
import sys

# --- 1. Geometry & Distribution Helpers ---

def tanh_distribution(n, delta_start, delta_end):
    """Generates 0..1 spacing with clustering at 0."""
    x = np.linspace(0, 1, n)
    b = 4.0 # Higher 'b' = tighter boundary layer
    return 1.0 + np.tanh(b * (x - 1.0)) / np.tanh(b)

class AirfoilGenerator:
    @staticmethod
    def naca4(number, n_points):
        """Standard NACA 4-digit generator."""
        m = int(number[0]) / 100.0
        p = int(number[1]) / 10.0
        t = int(number[2:]) / 100.0
        beta = np.linspace(0, np.pi, n_points)
        x = (1 - np.cos(beta)) / 2.0
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        yc, dyc_dx = np.zeros_like(x), np.zeros_like(x)
        
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
        return xu, yu, xl, yl

# --- 2. The Mesh Generator (TFI + Elliptic) ---

class StructuredMesher:
    def __init__(self, airfoil_coords, radius, wake_len, ni_wake, nj):
        self.xu, self.yu, self.xl, self.yl = airfoil_coords
        self.radius = radius
        self.wake_len = wake_len
        self.ni_wake = ni_wake
        self.nj = nj
        self.X = None
        self.Y = None
        self.ni = 0 # Will be set during generation

    def generate_algebraic(self):
        """Step 1: Generate initial TFI mesh."""
        # Wake Construction
        wake_x_lower = np.linspace(1.0 + self.wake_len, 1.0, self.ni_wake)
        wake_y_lower = np.zeros_like(wake_x_lower)
        wake_x_upper = np.linspace(1.0, 1.0 + self.wake_len, self.ni_wake)
        wake_y_upper = np.zeros_like(wake_x_upper)

        # Inner Boundary (Airfoil + Wake)
        inner_x = np.concatenate([wake_x_lower[:-1], self.xl[::-1][:-1], self.xu, wake_x_upper[1:]])
        inner_y = np.concatenate([wake_y_lower[:-1], self.yl[::-1][:-1], self.yu, wake_y_upper[1:]])
        
        self.ni = len(inner_x)
        
        # Outer Boundary (C-Shape)
        outer_x = np.zeros(self.ni)
        outer_y = np.zeros(self.ni)
        
        idx_wake_end_lower = self.ni_wake - 1
        idx_wake_start_upper = self.ni - self.ni_wake

        for i in range(self.ni):
            if i < idx_wake_end_lower:
                t = i / idx_wake_end_lower
                outer_x[i] = (1.0 + self.wake_len)*(1-t) + 1.0*t
                outer_y[i] = -self.radius
            elif i > idx_wake_start_upper:
                t = (i - idx_wake_start_upper) / (self.ni - 1 - idx_wake_start_upper)
                outer_x[i] = 1.0*(1-t) + (1.0 + self.wake_len)*t
                outer_y[i] = self.radius
            else:
                t_arc = (i - idx_wake_end_lower) / (idx_wake_start_upper - idx_wake_end_lower)
                angle = (3*np.pi/2) - t_arc * np.pi 
                outer_x[i] = 1.0 + self.radius * np.cos(angle)
                outer_y[i] = self.radius * np.sin(angle)

        # TFI Generation
        self.X = np.zeros((self.nj, self.ni))
        self.Y = np.zeros((self.nj, self.ni))
        dist = tanh_distribution(self.nj, 0, 1)

        for i in range(self.ni):
            for j in range(self.nj):
                s = dist[j]
                self.X[j, i] = inner_x[i] + s * (outer_x[i] - inner_x[i])
                self.Y[j, i] = inner_y[i] + s * (outer_y[i] - inner_y[i])
        
        print("Algebraic Mesh Generated.")

    def smooth_elliptic(self, iterations=50, freeze_layers=5):
        """
        Step 2: Solve Poisson Equation (Winslow Smoother).
        
        :param freeze_layers: Number of layers near wall to strictly PRESERVE (The Prism Layer).
                              Elliptic smoothing tends to wash out clustering, so we skip the BL.
        """
        print(f"Starting Elliptic Smoothing ({iterations} iterations)...")
        
        # Create copies to update
        X_new = self.X.copy()
        Y_new = self.Y.copy()
        
        for it in range(iterations):
            # Maximum error tracker
            max_diff = 0.0
            
            # Loop over INTERIOR nodes only
            # We start at 'freeze_layers' to preserve the prism layer
            for j in range(freeze_layers, self.nj - 1):
                for i in range(1, self.ni - 1):
                    
                    # Derivatives (Central Differences)
                    x_xi = 0.5 * (self.X[j, i+1] - self.X[j, i-1])
                    x_eta = 0.5 * (self.X[j+1, i] - self.X[j-1, i])
                    y_xi = 0.5 * (self.Y[j, i+1] - self.Y[j, i-1])
                    y_eta = 0.5 * (self.Y[j+1, i] - self.Y[j-1, i])
                    
                    # Coefficients for Winslow Generator (Orthogonality control)
                    alpha = x_eta**2 + y_eta**2
                    beta  = x_xi * x_eta + y_xi * y_eta
                    gamma = x_xi**2 + y_xi**2
                    
                    # Solving: alpha*r_xx - 2*beta*r_xy + gamma*r_yy = 0
                    # Discretized form (SOR/Gauss-Seidel step)
                    
                    x_next = (alpha * (self.X[j, i+1] + self.X[j, i-1]) + 
                              gamma * (self.X[j+1, i] + self.X[j-1, i]) - 
                              2 * beta * 0.25 * (self.X[j+1, i+1] - self.X[j-1, i+1] - self.X[j+1, i-1] + self.X[j-1, i-1])) / (2 * (alpha + gamma))
                              
                    y_next = (alpha * (self.Y[j, i+1] + self.Y[j, i-1]) + 
                              gamma * (self.Y[j+1, i] + self.Y[j-1, i]) - 
                              2 * beta * 0.25 * (self.Y[j+1, i+1] - self.Y[j-1, i+1] - self.Y[j+1, i-1] + self.Y[j-1, i-1])) / (2 * (alpha + gamma))
                    
                    # Relaxation (optional, set to 1.0 for standard)
                    omega = 0.5
                    X_new[j, i] = (1-omega)*self.X[j, i] + omega*x_next
                    Y_new[j, i] = (1-omega)*self.Y[j, i] + omega*y_next
                    
                    dist = np.sqrt((X_new[j, i] - self.X[j, i])**2 + (Y_new[j, i] - self.Y[j, i])**2)
                    if dist > max_diff: max_diff = dist
            
            # Update grid
            self.X = X_new.copy()
            self.Y = Y_new.copy()
            
            if it % 10 == 0:
                print(f"  Iter {it}: Max Move = {max_diff:.6f}")

    def save_to_file(self, filename):
        """Saves simple XY format."""
        with open(filename, 'w') as f:
            f.write(f"Dimensions: {self.ni} {self.nj}\n")
            f.write("X Y\n")
            for j in range(self.nj):
                for i in range(self.ni):
                    f.write(f"{self.X[j,i]:.6f} {self.Y[j,i]:.6f}\n")
        print(f"Mesh saved to {filename}")


# --- 3. Visualization ---

def plot_portfolio(mesher, freeze_layers):
    X, Y = mesher.X, mesher.Y
    nj, ni = X.shape
    
    fig = plt.figure(figsize=(14, 6))
    
    # --- Subplot 1: Full Mesh ---
    ax1 = fig.add_subplot(1, 2, 1)
    # Plot every 2nd line for clarity in full view
    for j in range(0, nj, 2): ax1.plot(X[j, :], Y[j, :], 'k-', lw=0.3, alpha=0.4)
    for i in range(0, ni, 4): ax1.plot(X[:, i], Y[:, i], 'k-', lw=0.3, alpha=0.4)
    
    # Boundaries
    ax1.plot(X[0, :], Y[0, :], 'r-', lw=1, label='Inner (Wall/Wake)')
    ax1.plot(X[-1, :], Y[-1, :], 'b-', lw=1, label='Farfield')
    ax1.set_title("Full C-Topology Mesh (Elliptic Smoothed)")
    ax1.set_aspect('equal')
    ax1.legend()

    # --- Subplot 2: Boundary Layer Zoom (Prism Layers) ---
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Determine indices for airfoil (excluding wake)
    idx_start = mesher.ni_wake - 5
    idx_end = ni - mesher.ni_wake + 5
    
    # Plot ONLY the prism layers (first 'freeze_layers' + a few more)
    view_layers = freeze_layers + 10
    
    # Plot Prism Layer Grid
    for j in range(view_layers):
        # Color the "Frozen" prism layers differently
        color = 'green' if j < freeze_layers else 'black'
        lw = 0.8 if j < freeze_layers else 0.4
        ax2.plot(X[j, idx_start:idx_end], Y[j, idx_start:idx_end], color=color, lw=lw)
        
    for i in range(idx_start, idx_end):
        ax2.plot(X[:view_layers, i], Y[:view_layers, i], 'k-', lw=0.3)

    # Highlight Surface
    ax2.plot(X[0, idx_start:idx_end], Y[0, idx_start:idx_end], 'r-', lw=2, label='Wall')
    
    # Annotation
    mid_x = X[freeze_layers, int(ni/2)]
    mid_y = Y[freeze_layers, int(ni/2)]
    ax2.text(mid_x, mid_y + 0.05, f"Prism Layer ({freeze_layers} cells)", 
             color='green', fontsize=10, fontweight='bold', ha='center')

    ax2.set_title("Zoom: Boundary Layer / Prism Layer")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect('equal')
    ax2.set_xlim(-0.1, 1.1) # Focus on Airfoil
    ax2.set_ylim(-0.2, 0.2)
    
    plt.tight_layout()
    plt.show()

# --- 4. Main Execution ---

if __name__ == "__main__":
    # Params
    naca = "4412"
    n_airfoil_points = 100
    wake_len = 5.0
    radius = 5.0
    ni_wake = 20 # Points in wake
    nj = 60      # Normal layers
    
    print("1. Generating Geometry...")
    xu, yu, xl, yl = AirfoilGenerator.naca4(naca, n_airfoil_points)
    
    print("2. Initializing Mesher...")
    mesher = StructuredMesher((xu, yu, xl, yl), radius, wake_len, ni_wake, nj)
    
    # Step A: Algebraic
    mesher.generate_algebraic()
    
    # Step B: Elliptic Smoothing with Prism Layer Protection
    # We freeze the first 10 layers to ensure the boundary layer isn't washed out
    mesher.smooth_elliptic(iterations=50, freeze_layers=10)
    
    # Step C: Save
    mesher.save_to_file("airfoil_mesh.xy")
    
    # Step D: Visualize
    plot_portfolio(mesher, freeze_layers=10)