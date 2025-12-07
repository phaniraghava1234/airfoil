import numpy as np
import matplotlib.pyplot as plt

# --- 1. Math Helpers ---

def tanh_distribution(n, val_start, val_end, b_cluster=4.0):
    """
    Generates a 1D distribution from val_start to val_end.
    Clusters points near val_start using hyperbolic tangent.
    """
    x = np.linspace(0, 1, n)
    # Tanh function: 1 + tanh(b*(x-1))/tanh(b) maps 0..1 to 0..1 with clustering at 0
    dist = 1.0 + np.tanh(b_cluster * (x - 1.0)) / np.tanh(b_cluster)
    return val_start + dist * (val_end - val_start)

def geometric_expansion(n_points, start_val, length, initial_dx):
    """
    Generates points starting at start_val with strictly controlled initial spacing.
    Expands geometrically to fill 'length'.
    """
    # We solve for the expansion ratio 'r' such that sum of geometric series = length
    # length = dx * (1 - r^(n-1)) / (1 - r)
    # This requires a root finder, but for this portfolio we can approximate
    # or just let the length float slightly to match the physics requirements.
    
    # Simple approach: Fixed expansion ratio (e.g., 1.1) 
    # and we generate points until we exceed length or hit n_points.
    
    vals = [start_val]
    dx = initial_dx
    r = 1.15 # Expansion ratio (15% growth per cell)
    
    for _ in range(n_points - 1):
        vals.append(vals[-1] + dx)
        dx *= r
    
    # Scale to force exact length match (optional, but keeps domain size consistent)
    vals = np.array(vals)
    actual_len = vals[-1] - vals[0]
    scale = length / actual_len
    vals = start_val + (vals - start_val) * scale
    return vals

# --- 2. Geometry Generators ---

class AirfoilGenerator:
    @staticmethod
    def naca4(number, n_points):
        """Generates NACA 4-digit airfoil coordinates (Cosine Spacing)."""
        m = int(number[0]) / 100.0
        p = int(number[1]) / 10.0
        t = int(number[2:]) / 100.0
        
        # Cosine clustering for LE and TE resolution
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
                yc[i] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x[i] - x[i]**2)
                dyc_dx[i] = (2 * m / (1 - p)**2) * (p - x[i])
                
        theta = np.arctan(dyc_dx)
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)
        
        return xu, yu, xl, yl

# --- 3. Structured Mesher (The Core Logic) ---

class CMeshGenerator:
    def __init__(self, airfoil_coords, radius, wake_len, ni_wake, nj):
        self.xu, self.yu, self.xl, self.yl = airfoil_coords
        self.radius = radius
        self.wake_len = wake_len
        self.ni_wake = ni_wake
        self.nj = nj
        self.X = None
        self.Y = None

    def generate_algebraic(self, wake_type="Straight"):
        """
        Generates the initial Algebraic Mesh using TFI.
        wake_type: "Straight" (horizontal) or "Angle" (follows mean camber)
        """
        
        # 1. Calculate Trailing Edge Spacing from Airfoil Points
        # Distance between last two points on upper surface
        dx_te_upper = np.sqrt((self.xu[-1] - self.xu[-2])**2 + (self.yu[-1] - self.yu[-2])**2)
        dx_te_lower = np.sqrt((self.xl[-1] - self.xl[-2])**2 + (self.yl[-1] - self.yl[-2])**2)
        avg_dx_te = (dx_te_upper + dx_te_lower) / 2.0
        
        # 2. Generate Wake Points (Matched Spacing)
        # We start at x=1.0 and go downstream
        wake_x_vals = geometric_expansion(self.ni_wake, 1.0, self.wake_len, avg_dx_te)
        
        # Wake Y values
        if wake_type == "Straight":
            wake_y_vals = np.zeros_like(wake_x_vals) # Flat wake at y=0
        else:
            # Simple fallback: assumes exit angle 0 for now. 
            # Real code would calculate bisector angle at TE.
            wake_y_vals = np.zeros_like(wake_x_vals)

        # 3. Assemble Inner Boundary (J=0)
        # Order: Lower Wake (Far->TE) -> Lower Surf (TE->LE) -> Upper Surf (LE->TE) -> Upper Wake (TE->Far)
        
        wake_x_lower = wake_x_vals[::-1] # Flip for incoming direction
        wake_y_lower = wake_y_vals[::-1]
        
        # Splice arrays. Note: [:-1] avoids duplicating the connector points
        inner_x = np.concatenate([wake_x_lower[:-1], self.xl[::-1][:-1], self.xu, wake_x_vals[1:]])
        inner_y = np.concatenate([wake_y_lower[:-1], self.yl[::-1][:-1], self.yu, wake_y_vals[1:]])
        
        self.ni = len(inner_x)
        
        # 4. Assemble Outer Boundary (Farfield)
        # Map indices to C-Shape geometry
        outer_x = np.zeros(self.ni)
        outer_y = np.zeros(self.ni)
        
        idx_wake_end_lower = self.ni_wake - 1
        idx_wake_start_upper = self.ni - self.ni_wake
        
        for i in range(self.ni):
            if i < idx_wake_end_lower:
                # Lower Wake Boundary
                t = i / idx_wake_end_lower
                # Match x-coordinates of wake for better orthogonality
                outer_x[i] = inner_x[i] 
                outer_y[i] = -self.radius
                
            elif i > idx_wake_start_upper:
                # Upper Wake Boundary
                outer_x[i] = inner_x[i]
                outer_y[i] = self.radius
                
            else:
                # C-Arc around Airfoil
                t_arc = (i - idx_wake_end_lower) / (idx_wake_start_upper - idx_wake_end_lower)
                angle = (3 * np.pi / 2) - t_arc * np.pi
                outer_x[i] = 1.0 + self.radius * np.cos(angle)
                outer_y[i] = self.radius * np.sin(angle)

        # 5. TFI Generation (Transfinite Interpolation)
        self.X = np.zeros((self.nj, self.ni))
        self.Y = np.zeros((self.nj, self.ni))
        
        # Distribution normal to wall
        # b_cluster=4.5 ensures tight prism layer
        dist = tanh_distribution(self.nj, 0, 1, b_cluster=4.5)

        for i in range(self.ni):
            for j in range(self.nj):
                s = dist[j]
                self.X[j, i] = inner_x[i] + s * (outer_x[i] - inner_x[i])
                self.Y[j, i] = inner_y[i] + s * (outer_y[i] - inner_y[i])

# --- 4. Elliptic Smoother (Refinement) ---

class EllipticSmoother:
    @staticmethod
    def smooth(X, Y, n_iter=100, relaxation=0.2, locked_layers=10):
        """
        Winslow Elliptic Smoother with Boundary Layer Protection.
        """
        nj, ni = X.shape
        X_new = X.copy()
        Y_new = Y.copy()

        for k in range(n_iter):
            X_old = X_new.copy()
            Y_old = Y_new.copy()

            # Iterate over interior nodes, skipping locked prism layers
            for j in range(locked_layers, nj - 1):
                for i in range(1, ni - 1):
                    
                    x_xi = 0.5 * (X_old[j, i+1] - X_old[j, i-1])
                    x_eta = 0.5 * (X_old[j+1, i] - X_old[j-1, i])
                    y_xi = 0.5 * (Y_old[j, i+1] - Y_old[j, i-1])
                    y_eta = 0.5 * (Y_old[j+1, i] - Y_old[j-1, i])
                    
                    alpha = x_eta**2 + y_eta**2
                    gamma = x_xi**2 + y_xi**2
                    beta  = x_xi * x_eta + y_xi * y_eta # Cross term for non-orthogonality
                    
                    # Poisson solver step
                    val_x = (alpha * (X_old[j, i+1] + X_old[j, i-1]) + 
                             gamma * (X_old[j+1, i] + X_old[j-1, i]) -
                             2*beta*0.25*(X_old[j+1,i+1] - X_old[j-1,i+1] - X_old[j+1,i-1] + X_old[j-1,i-1])) / (2 * (alpha + gamma))
                    
                    val_y = (alpha * (Y_old[j, i+1] + Y_old[j, i-1]) + 
                             gamma * (Y_old[j+1, i] + Y_old[j-1, i]) -
                             2*beta*0.25*(Y_old[j+1,i+1] - Y_old[j-1,i+1] - Y_old[j+1,i-1] + Y_old[j-1,i-1])) / (2 * (alpha + gamma))

                    X_new[j, i] = (1 - relaxation) * X_old[j, i] + relaxation * val_x
                    Y_new[j, i] = (1 - relaxation) * Y_old[j, i] + relaxation * val_y
                    
        return X_new, Y_new

# --- 5. Main Execution ---

def main():
    # Setup
    naca = "4412"
    n_airfoil_points = 80 # High resolution for surface
    wake_len = 8.0
    radius = 6.0
    ni_wake = 30 # Points in wake
    nj = 60      # Normal layers (high for BL resolution)
    locked_layers = 15 # Protect first 15 layers from smoothing
    
    print(f"Generating mesh for NACA {naca}...")
    
    # 1. Geometry
    xu, yu, xl, yl = AirfoilGenerator.naca4(naca, n_airfoil_points)
    
    # 2. Mesher Initialization
    mesher = CMeshGenerator((xu, yu, xl, yl), radius, wake_len, ni_wake, nj)
    
    # 3. Generate Algebraic (with matched wake spacing)
    mesher.generate_algebraic(wake_type="Straight")
    
    # 4. Smooth
    print("Smoothing mesh (preserving boundary layer)...")
    X_smooth, Y_smooth = EllipticSmoother.smooth(mesher.X, mesher.Y, n_iter=150, locked_layers=locked_layers)
    
    # 5. Visualize
    print("Plotting...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot Full Mesh (Subsampled for clarity)
    for j in range(0, nj, 2):
        ax.plot(X_smooth[j, :], Y_smooth[j, :], 'k-', lw=0.3, alpha=0.3)
    for i in range(0, mesher.ni, 2):
        ax.plot(X_smooth[:, i], Y_smooth[:, i], 'k-', lw=0.3, alpha=0.3)
        
    # Plot Prism Layers (Green)
    for j in range(locked_layers):
        ax.plot(X_smooth[j, :], Y_smooth[j, :], 'g-', lw=0.5, alpha=0.6)
        
    # Plot Boundaries
    n_wake = ni_wake
    idx_start = n_wake - 1
    idx_end = mesher.ni - n_wake
    
    ax.plot(X_smooth[0, idx_start:idx_end+1], Y_smooth[0, idx_start:idx_end+1], 'r-', lw=2, label=f'NACA {naca}')
    ax.plot(X_smooth[0, :idx_start+1], Y_smooth[0, :idx_start+1], 'b--', label='Wake')
    ax.plot(X_smooth[0, idx_end:], Y_smooth[0, idx_end:], 'b--')

    # Zoom Inset for TE Quality Check
    axins = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
    # Show full density in zoom
    for j in range(30):
        color = 'g-' if j < locked_layers else 'k-'
        axins.plot(X_smooth[j, :], Y_smooth[j, :], color, lw=0.5)
    for i in range(idx_end - 10, idx_end + 10):
        axins.plot(X_smooth[:, i], Y_smooth[:, i], 'k-', lw=0.5)
        
    axins.set_xlim(0.9, 1.3)
    axins.set_ylim(-0.15, 0.15)
    axins.set_title("Trailing Edge Match (Zoom)")
    ax.indicate_inset_zoom(axins, edgecolor="black")

    ax.set_title(f"Refined C-Mesh for NACA {naca} with Prism Layer Protection")
    ax.set_aspect('equal')
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()