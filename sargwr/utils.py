import numpy as np

def add_intercept(X):
    '''
    Add an intercept column to the design matrix.
    '''
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if not np.allclose(X[:, 0], 1):
        X = np.column_stack((np.ones(X.shape[0]), X))

    return X

def get_W(W, Wis, use_coords, n):
    '''
    Validating and constructing W
    '''
    if use_coords:
        W = np.array([np.diag(Wi) for Wi in Wis])
    else:
        if W is None:
            raise ValueError('Provide W or set use_coords to True')
        else:
            if hasattr(W, 'full'):
                W = W.full()[0]
            else:
                W = np.asarray(W)
        if W.shape != (n, n):
            raise ValueError('W must have shape (n, n).')
    
    return W

def validate_bandwidth(bandwidth, fixed, n, p, multi=False):
    '''
    Validating bandwidth(s)
    '''
    if multi:
        bandwidth = np.asarray(bandwidth)
        if bandwidth.ndim != 1:
            raise ValueError('bandwidths must be a one-dimensional sequence.')
        if len(bandwidth) != p:
            raise ValueError(
                'A bandwidth must be supplied for every regressor. '
                f'Expected {p} bandwidths, '
                f'but received {len(bandwidth)}.'
            )
        if not np.all(np.isfinite(bandwidth)):
            raise ValueError('All bandwidths must be finite.')
        if np.any(bandwidth <= 0):
            raise ValueError('All bandwidths must be positive.')
        if not fixed:
            bandwidth = np.asarray(np.round(bandwidth), dtype=int)
            if np.any(bandwidth < 1):
                raise ValueError('Adaptive bandwidths must be positive integers.')
            if np.any(bandwidth >= n):
                raise ValueError(
                    'Adaptive bandwidths must be smaller than the number of observations.'
                )
    else:
        if not np.isfinite(bandwidth):
            raise ValueError('Bandwidth must be finite.')
        if bandwidth <= 0:
            raise ValueError('Bandwidth must be positive.')
        if not fixed:
            bandwidth = np.round(bandwidth)
            if bandwidth < 1:
                raise ValueError('Adaptive bandwidth must be positive integer.')
            if bandwidth >= n:
                raise ValueError(
                    'Adaptive bandwidth must be smaller than the number of observations.'
                )
    return bandwidth

def get_variable_names(variable_names, p, intercept):
    if variable_names is None:
        variable_names = [f'X{i}' for i in range(p)]
    else:
        variable_names = list(variable_names)
        if intercept:
            variable_names.insert(0, 'Intercept')
        if len(variable_names) != p:
            raise ValueError('variable_names must contain one name for every regressor.')
    return variable_names