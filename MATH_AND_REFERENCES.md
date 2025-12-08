# Mathematical Formulations and Code Correspondence

This document maps the mathematical theories used in this project to their specific implementations in the Python code.

## 1. NACA 4-Digit Series Airfoil
**Source**: *Abbott, I. H., & Von Doenhoff, A. E. (1959). Theory of wing sections.*

The thickness distribution $y_t$ is given by:
$$ y_t = 5t \left( 0.2969\sqrt{x} - 0.1260x - 0.3516x^2 + 0.2843x^3 - 0.1015x^4 \right) $$

**Code Implementation**: 
- File: `initial_tfi_2.py`
- Function: `AirfoilGenerator.naca4`
- Line: `yt = 5 * t * (0.2969 * np.sqrt(x) ...)`

## 2. Transfinite Interpolation (TFI)
**Source**: *Thompson, J. F., et al. (1985). Numerical grid generation.*

For a 2D domain with coordinates $(\xi, \eta)$ normalized to $[0,1]$, the linear TFI is defined as the Boolean sum of unidirectional interpolations. In this simplified C-mesh implementation, we interpolate primarily between the inner boundary (airfoil/wake) and outer boundary (farfield):

$$ \vec{r}(\xi, \eta) = (1 - \eta)\vec{r}_{inner}(\xi) + \eta \vec{r}_{outer}(\xi) $$

**Code Implementation**:
- File: `initial_tfi_2.py`
- Function: `TFIGenerator.generate`
- Line: `self.X[j,i] = (1-s)*inner_x[i] + s*outer_x[i]`

## 3. Elliptic Smoothing (Winslow's Method)
**Source**: *Winslow, A. M. (1966). Journal of Computational Physics.*

To optimize the mesh smoothness, we solve the Laplace equations for the curvilinear coordinates $(\xi, \eta)$ in physical space $(x, y)$:
$$ \nabla^2 \xi = 0, \quad \nabla^2 \eta = 0 $$

Inverting this to solve for $x(\xi, \eta)$ and $y(\xi, \eta)$ yields the non-linear elliptic system:
$$ \alpha \frac{\partial^2 \vec{r}}{\partial \xi^2} - 2\beta \frac{\partial^2 \vec{r}}{\partial \xi \partial \eta} + \gamma \frac{\partial^2 \vec{r}}{\partial \eta^2} = 0 $$

Where the coefficients are metric terms:
$$ \alpha = x_\eta^2 + y_\eta^2 $$
$$ \gamma = x_\xi^2 + y_\xi^2 $$
$$ \beta = x_\xi x_\eta + y_\xi y_\eta $$

**Code Implementation**:
- File: `tfi_finaly.py`
- Function: `winslow_smoother`
- Coefficients Calculation: 
  ```python
  alpha = x_eta**2 + y_eta**2
  gamma = x_xi**2 + y_xi**2
  beta  = x_xi * x_eta + y_xi * y_eta