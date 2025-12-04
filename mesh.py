import numpy as np
import matplotlib.pyplot as plt

# --- [Keep the AirfoilGenerator and tanh_distribution classes/functions exactly as before] ---
# (Re-pasting them here for a complete runnable script)

def tanh_distribution(n, delta_start, delta_end):
    x = np.linspace(0, 1, n)
    b = 3.0 # Increased clustering for better visualization
    return 1.0 + np.tanh(b * (x - 1.0)) / np.tanh(b)

class AirfoilGenerator:
    @staticmethod
    def naca4(number, n_points):
        m = int(number[0]) / 100.0
        p = int(number[1]) / 10.0
        t = int(number[2:]) / 100.0
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
        return xu, yu, xl, yl

class CMeshGenerator:
    def __init__(self, airfoil_coords, domain_radius, wake_length, ni, nj):
        self.xu, self.yu, self.xl, self.yl = airfoil_coords
        self.radius = domain_radius
        self.wake_len = wake_length
        self.ni = ni
        self.nj = nj
        self.n_wake = 20 # Store this so we can access it for plotting

    def generate(self):
        # 1. Inner Boundary
        wake_x_lower = np.linspace(1.0 + self.wake_len, 1.0, self.n_wake)
        wake_y_lower = np.zeros_like(wake_x_lower)
        wake_x_upper = np.linspace(1.0, 1.0 + self.wake_len, self.n_wake)
        wake_y_upper = np.zeros_like(wake_x_upper)

        inner_x = np.concatenate([
            wake_x_lower[:-1],
            self.xl[::-1][:-1],
            self.xu,
            wake_x_upper[1:]
        ])
        inner_y = np.concatenate([
            wake_y_lower[:-1],
            self.yl[::-1][:-1],
            self.yu,
            wake_y_upper[1:]
        ])

        n_points = len(inner_x)

        # 2. Outer Boundary
        outer_x = np.zeros(n_points)
        outer_y = np.zeros(n_points)
        
        idx_wake_end_lower = self.n_wake - 1
        idx_wake_start_upper = n_points - self.n_wake

        for i in range(n_points):
            if i < idx_wake_end_lower:
                # Lower Wake (Straight)
                t_local = i / idx_wake_end_lower
                outer_x[i] = (1.0 + self.wake_len) * (1 - t_local) + (1.0) * t_local
                outer_y[i] = -self.radius
            elif i > idx_wake_start_upper:
                # Upper Wake (Straight)
                t_local = (i - idx_wake_start_upper) / (n_points - 1 - idx_wake_start_upper)
                outer_x[i] = (1.0) * (1 - t_local) + (1.0 + self.wake_len) * t_local
                outer_y[i] = self.radius
            else:
                # C-Arc
                t_arc = (i - idx_wake_end_lower) / (idx_wake_start_upper - idx_wake_end_lower)
                # Map 0..1 to 270 deg down to 90 deg
                rad_angle = (3 * np.pi / 2) - t_arc * np.pi 
                outer_x[i] = 1.0 + self.radius * np.cos(rad_angle)
                outer_y[i] = self.radius * np.sin(rad_angle)

        # 3. TFI Generation
        X = np.zeros((self.nj, n_points))
        Y = np.zeros((self.nj, n_points))
        dist = tanh_distribution(self.nj, 0, 1)

        for i in range(n_points):
            for j in range(self.nj):
                s = dist[j]
                X[j, i] = inner_x[i] + s * (outer_x[i] - inner_x[i])
                Y[j, i] = inner_y[i] + s * (outer_y[i] - inner_y[i])

        return X, Y

# --- Execution and Visualization ---
def main():
    print("Generating NACA 2412 Geometry...")
    xu, yu, xl, yl = AirfoilGenerator.naca4("2412", n_points=50)

    print("Generating C-Mesh...")
    # Define dimensions
    radius = 5.0
    wake_len = 6.0
    
    mesher = CMeshGenerator(
        airfoil_coords=(xu, yu, xl, yl),
        domain_radius=radius,
        wake_length=wake_len,
        ni=0,
        nj=25
    )

    X, Y = mesher.generate()

    print("Plotting...")
    plt.figure(figsize=(12, 8))
    
    # 1. Plot the Grid Lines
    nj, ni = X.shape
    for j in range(nj):
        plt.plot(X[j, :], Y[j, :], 'k-', linewidth=0.3, alpha=0.4)
    for i in range(ni):
        plt.plot(X[:, i], Y[:, i], 'k-', linewidth=0.3, alpha=0.4)

    # 2. Plot Boundaries clearly
    
    # Calculate indices for the airfoil part vs wake part
    n_wake = mesher.n_wake
    idx_start_airfoil = n_wake - 1
    idx_end_airfoil = ni - n_wake
    
    # Plot Wake Line (Centerline downstream)
    # Lower part of wake line
    plt.plot(X[0, :idx_start_airfoil+1], Y[0, :idx_start_airfoil+1], 'k--', linewidth=1.0, label='Wake Line')
    # Upper part of wake line
    plt.plot(X[0, idx_end_airfoil:], Y[0, idx_end_airfoil:], 'k--', linewidth=1.0)

    # Plot Airfoil Surface (Only the middle section of j=0)
    plt.plot(X[0, idx_start_airfoil:idx_end_airfoil+1], 
             Y[0, idx_start_airfoil:idx_end_airfoil+1], 
             'r-', linewidth=2.5, label='Airfoil Surface')
    
    # Plot Farfield (Blue Line) - The last j index
    plt.plot(X[-1, :], Y[-1, :], 'b-', linewidth=2.0, label='Farfield Boundary')

    plt.title(f"C-Topology Structured Mesh (NACA 2412)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal') # This ensures circles look like circles
    plt.legend(loc='upper right')
    
    # Optional: Add a zoomed-in inset to show the nose detail
    # (Advanced touch for portfolio)
    ax = plt.gca()
    axins = ax.inset_axes([0.05, 0.55, 0.3, 0.3])
    for j in range(nj):
        axins.plot(X[j, :], Y[j, :], 'k-', linewidth=0.3, alpha=0.4)
    for i in range(ni):
        axins.plot(X[:, i], Y[:, i], 'k-', linewidth=0.3, alpha=0.4)
    axins.plot(X[0, idx_start_airfoil:idx_end_airfoil+1], 
               Y[0, idx_start_airfoil:idx_end_airfoil+1], 'r-', linewidth=2.5)
    axins.set_xlim(-0.1, 0.4)
    axins.set_ylim(-0.2, 0.2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.set_title("Leading Edge Detail")
    ax.indicate_inset_zoom(axins, edgecolor="black")

    plt.show()

if __name__ == "__main__":
    main()