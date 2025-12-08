import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
# Ensure initial_tfi_2.py is in the same directory
from initial_tfi_2 import AirfoilGenerator, TFIGenerator, save_xyz 

def winslow_smoother(X, Y, iterations=100, freeze_layers=10, omega=0.5):
    """
    Elliptic smoothing using Winslow's equations (Laplace equation in computational space).
    
    References:
    1. Winslow, A. M. (1966). 'Numerical solution of the quasilinear poisson equation in a nonuniform triangle mesh'.
       Journal of Computational Physics.
    2. Spekreijse, S. P. (1995). 'Elliptic Grid Generation based on Laplace Equations and Algebraic Transformations'.
    
    Parameters:
    - freeze_layers: Number of layers near the wall to 'freeze' to preserve TFI orthogonality.
    - omega: Relaxation factor (SOR method).
    """
    nj, ni = X.shape
    X_new = X.copy()
    Y_new = Y.copy()
    
    print(f"Smoothing for {iterations} iterations (Locked Layers: {freeze_layers})...")
    
    for _ in range(iterations):
        X_old = X_new.copy()
        Y_old = Y_new.copy()
        
        # Iterate over internal nodes only
        for j in range(freeze_layers, nj - 1):
            for i in range(1, ni - 1):
                
                # Metrics of the transformation (Finite Difference approximation)
                x_xi  = 0.5 * (X_old[j, i+1] - X_old[j, i-1])
                x_eta = 0.5 * (X_old[j+1, i] - X_old[j-1, i])
                y_xi  = 0.5 * (Y_old[j, i+1] - Y_old[j, i-1])
                y_eta = 0.5 * (Y_old[j+1, i] - Y_old[j-1, i])
                
                # Coefficients for the elliptic equation (g_11, g_22, g_12)
                # alpha * r_xx - 2*beta * r_xy + gamma * r_yy = 0
                alpha = x_eta**2 + y_eta**2
                gamma = x_xi**2 + y_xi**2
                beta  = x_xi * x_eta + y_xi * y_eta
                
                denom = 2 * (alpha + gamma)
                if denom == 0: continue
                
                # Solve for position at (i,j)
                x_next = (alpha * (X_old[j, i+1] + X_old[j, i-1]) +
                           gamma * (X_old[j+1, i] + X_old[j-1, i]) -
                           2 * beta * 0.25 * (X_old[j+1,i+1] - X_old[j-1,i+1] - X_old[j+1,i-1] + X_old[j-1,i-1])) / denom
                
                y_next = (alpha * (Y_old[j, i+1] + Y_old[j, i-1]) +
                           gamma * (Y_old[j+1, i] + Y_old[j-1, i]) -
                           2 * beta * 0.25 * (Y_old[j+1,i+1] - Y_old[j-1,i+1] - Y_old[j+1,i-1] + Y_old[j-1,i-1])) / denom
                
                # Successive Over-Relaxation (SOR) update
                X_new[j, i] = (1 - omega) * X_old[j, i] + omega * x_next
                Y_new[j, i] = (1 - omega) * Y_old[j, i] + omega * y_next
                
    return X_new, Y_new

def plot_comparison(X0, Y0, X1, Y1, freeze_layers):
    """Simple side-by-side comparison of Global Mesh topology."""
    fig, (ax1, ax2) = plt.subplots(1, 2, num=1, figsize=(16, 6))
    
    # 1. Algebraic
    ax1.plot(X0.T, Y0.T, 'k-', lw=0.3, alpha=0.3)
    ax1.plot(X0, Y0, 'k-', lw=0.3, alpha=0.3)
    ax1.set_title("Initial Algebraic (TFI)")
    ax1.set_xlim(-2, 3); ax1.set_ylim(-2, 2); ax1.set_aspect('equal')
    
    # 2. Elliptic
    nj, ni = X1.shape
    # Plot Prism Layers (Green)
    for i in range(0, ni, 2):
        ax2.plot(X1[:freeze_layers+1, i], Y1[:freeze_layers+1, i], 'g-', lw=0.6)
    for j in range(freeze_layers+1):
        ax2.plot(X1[j, :], Y1[j, :], 'g-', lw=0.6)
        
    # Plot Smoothed Field (Black)
    for i in range(0, ni, 2):
        ax2.plot(X1[freeze_layers:, i], Y1[freeze_layers:, i], 'k-', lw=0.3, alpha=0.3)
    for j in range(freeze_layers, nj, 2):
        ax2.plot(X1[j, :], Y1[j, :], 'k-', lw=0.3, alpha=0.3)

    ax2.plot(X1[0,:], Y1[0,:], 'r-', lw=2.0)
    ax2.set_title("Elliptic Smoothed")
    ax2.set_xlim(-2, 3); ax2.set_ylim(-2, 2); ax2.set_aspect('equal')

def plot_prism_showcase(X, Y, freeze_layers):
    """
    Detailed plot focused on the Airfoil with a zoomed INSET for the Leading Edge.
    """
    fig, ax = plt.subplots(num=2, figsize=(14, 10))
    nj, ni = X.shape
    
    # --- 1. Main Plot (Focus on Airfoil) ---
    
    # Plot Prism Layers (Green)
    for i in range(ni):
        ax.plot(X[:freeze_layers+1, i], Y[:freeze_layers+1, i], color='#00aa00', lw=0.8)
    for j in range(freeze_layers+1):
        ax.plot(X[j, :], Y[j, :], color='#00aa00', lw=0.8)
        
    # Plot Outer Smoothed Mesh (Black, thinner)
    step = 1
    for i in range(0, ni, step):
        ax.plot(X[freeze_layers:, i], Y[freeze_layers:, i], 'k-', lw=0.4, alpha=0.4)
    for j in range(freeze_layers, nj, step):
        ax.plot(X[j, :], Y[j, :], 'k-', lw=0.4, alpha=0.4)
        
    # Plot Wall
    ax.plot(X[0, :], Y[0, :], 'r-', lw=2.5, label='NACA 4412 Wall')
    
    ax.set_title("Final Mesh: Boundary Layer Preservation & Elliptic Smoothing", fontsize=14)
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.6, 0.6)
    ax.legend(loc='lower right')

    # --- 2. Zoomed Inset (Leading Edge / Nose) ---
    axins = inset_axes(ax, width="35%", height="35%", loc='upper left', borderpad=2)
    
    nose_idx = ni // 2
    idx_range = 15 
    i_s, i_e = nose_idx - idx_range, nose_idx + idx_range
    
    # Plot Prism (Green) inside inset
    for i in range(i_s, i_e):
        axins.plot(X[:freeze_layers+1, i], Y[:freeze_layers+1, i], color='#00aa00', lw=1.0)
    for j in range(freeze_layers+2): 
        color = '#00aa00' if j <= freeze_layers else 'k'
        lw = 1.0 if j <= freeze_layers else 0.5
        axins.plot(X[j, i_s:i_e], Y[j, i_s:i_e], color=color, lw=lw)
        
    # Plot Outer (Black) inside inset
    for i in range(i_s, i_e):
        axins.plot(X[freeze_layers:, i], Y[freeze_layers:, i], 'k-', lw=0.5, alpha=0.5)
    
    # Plot Wall in inset
    axins.plot(X[0, i_s:i_e], Y[0, i_s:i_e], 'r-', lw=3.0)
    
    axins.set_title("Leading Edge Zoom", fontsize=10)
    axins.set_xlim(-0.02, 0.05) 
    axins.set_ylim(-0.03, 0.03)
    axins.set_xticks([]) 
    axins.set_yticks([])
    
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

if __name__ == "__main__":
    naca = "4412"
    ni_wake = 30
    nj = 60
    freeze = 15
    
    # 1. Generate Algebraic Grid
    xu, yu, xl, yl = AirfoilGenerator.naca4(naca, 80)
    gen = TFIGenerator(xu, yu, xl, yl, wake_len=6.0, ni_wake=ni_wake, nj=nj, clustering=4.5)
    X_alg, Y_alg = gen.generate()
    
    # 2. Apply Elliptic Smoothing
    X_ell, Y_ell = winslow_smoother(X_alg, Y_alg, iterations=100, freeze_layers=freeze)
    
    # 3. Save
    save_xyz(X_ell, Y_ell, "final_mesh.csv")

    # 4. Visualization
    plot_comparison(X_alg, Y_alg, X_ell, Y_ell, freeze) 
    plot_prism_showcase(X_ell, Y_ell, freeze)          

    plt.show()