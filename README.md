# SARGWR

Spatial Autoregressive Geographically Weighted Regression (SARGWR) implemented in Python.

SARGWR combines the advantages of the Spatial Autoregressive (SAR) model and Geographically Weighted Regression (GWR) by accounting for both spatial dependence in the response variable and spatial heterogeneity in regression coefficients.

The current Geographically Weighted Regression - Saptial Lag model is defined as

\[
y = \rho Wy + X\beta(u,v) + \varepsilon,
\]

where

- \(y\) is the response variable,
- \(W\) is the spatial weights matrix,
- \(\rho\) is the spatial autoregressive parameter,
- \(X\) is the matrix of explanatory variables,
- \(\beta(u,v)\) are location-specific regression coefficients,
- \(\varepsilon\) is the random error.

---

## Features

- Spatial autoregressive geographically weighted regression
- Fixed and adaptive bandwidths
- Multiple kernel functions
    - Gaussian
    - Bisquare
    - Triangular
    - Exponential
    - Uniform
    - Quadratic
    - Quartic
- Automatic bandwidth selection
    - Golden section search
    - Interval search
- Euclidean and spherical distances
- Parallel estimation
- Model diagnostics
    - RSS
    - R²
    - Adjusted R²
    - AIC
    - AICc
    - BIC
    - Log-likelihood
    - Effective degrees of freedom
- Local coefficient estimates
- Local t-values
- Local significance filtering

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/sargwr.git
```

Install dependencies

```bash
pip install numpy scipy joblib
```

---

## Example

```python
from sargwr import GWR_SL

model = GWR_SL(kernel='bisquare')
model.fit(coords, y, X, W=W)

print(model.summary())
```

---

## Bandwidth Selection

Automatic bandwidth selection can be performed using

```python
selector='golden'
```

or

```python
selector='interval'
```

The optimization criterion may be

- AIC
- AICc
- BIC

---

## References

Brunsdon, C., Fotheringham, A. S., & Charlton, M. (1996). Geographically Weighted Regression.

Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002). Geographically Weighted Regression.

Anselin, L. (1988). Spatial Econometrics: Methods and Models.

---

## License

MIT License.
