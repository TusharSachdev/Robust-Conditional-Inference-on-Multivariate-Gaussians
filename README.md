# Robust-Conditional-Inference-on-Multivariate-Gaussians

**Objective:** Explore how small perturbations or model misspecifications affect conditional inference in Gaussian graphical models.

---

## Project Overview

This project investigates the **robustness of conditional inference** in multivariate Gaussian settings. Specifically, we study how:

- Small global perturbations affect precision matrix estimates.
- Individual variable perturbations impact the model (variable fragility).
- Correlated perturbations influence inference outcomes.

The empirical findings are inspired by Alexander Fisher’s theoretical work on the **indiscriminate disruption of conditional inference**, highlighting where standard Gaussian inference fails.

---

## Methodology

1. **Data Generation**
   - Generated multivariate Gaussian data with 5 variables and 500 samples.
   - Controlled correlation structure to simulate realistic dependencies.

2. **Precision Matrix Estimation**
   - Used `GraphicalLasso` (from `scikit-learn`) for sparse precision estimation.
   - Evaluated baseline precision matrix.

3. **Perturbation Analysis**
   - **Global perturbations:** Added random noise scaled by `ε` to all variables.
   - **Variable-wise perturbations:** Perturbed each variable individually to calculate fragility scores.
   - **Correlated perturbations:** Applied structured perturbations preserving correlation patterns.

4. **Fragility Measurement**
   - Calculated changes in precision matrix norms.
   - Assigned **fragility scores** to variables: higher scores indicate more critical variables for model stability.

5. **Visualization**
   - Plotted **global perturbation vs change in precision**.
   - Bar plot of **variable-wise fragility scores**.
   - Annotated critical variables with numerical fragility scores.

---

## Key Results

### Estimated Precision Matrix
[[ 2.0998 -1.4849 0. 0. 0. ]
[-1.4849 2.1634 -0.3473 0.0730 0. ]
[ 0. -0.3473 1.4028 -0.4926 0.0872]
[ 0. 0.0730 -0.4926 1.5498 -0.8141]
[ 0. 0. 0.0872 -0.8141 1.4425]]


### Global Perturbation (ε vs Change in Precision)
| Epsilon | Change in Precision |
|---------|------------------|
| 0.01    | 0.0001           |
| 0.05    | 0.0005           |
| 0.1     | 0.0008           |
| 0.2     | 0.0020           |
| 0.3     | 0.0034           |
| 0.5     | 0.0047           |

### Variable-wise Fragility Scores
| Variable | Fragility Score | Relative |
|----------|----------------|---------|
| 0        | 0.319          | 1.00    |
| 1        | 0.248          | 0.78    |
| 3        | 0.217          | 0.68    |
| 2        | 0.138          | 0.43    |
| 4        | 0.077          | 0.24    |

**Most critical variable:** Variable 0  
**Least critical variable:** Variable 4

---

## Python Libraries Used

- `numpy` — data generation and numerical computations  
- `scikit-learn` — `GraphicalLasso` for precision estimation  
- `matplotlib` — visualization of perturbation effects  
- `pandas` — optional, for data handling

---

## Insights

- **Fragility is heterogeneous:** Some variables disproportionately influence model stability.  
- **Global perturbations** affect the precision matrix gradually, while **variable-wise perturbations** reveal hidden vulnerabilities.  
- **Correlated perturbations** have minor effects for small ε, demonstrating that robustness depends on perturbation structure.

This analysis provides a practical demonstration of where **standard Gaussian conditional inference may fail**, aligning with the theoretical insights of Alexander Fisher.

---
