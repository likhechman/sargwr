__author__ = 'Alexandra Kaluzhak'

import numpy as np
from scipy.optimize import minimize_scalar

from joblib import Parallel, delayed

def get_y_star(y, W, rho):
    '''
    Estimate y_star using (I - rho W) y.
    '''
    I = np.identity(y.shape[0])
    y_star = np.dot((I - rho * W), y)
    return y_star

def estimate_gwr_betas(y, X, Wi):
    '''
    Estimate local GWR betas.
    '''
    XtWi = np.dot(X.T, Wi)
    XtWiX = np.dot(XtWi, X)
    XtWiX_inv = np.linalg.solve(XtWiX, np.identity(X.shape[1]))
    XtWiX_XtWi = np.linalg.solve(XtWiX, XtWi)
    betas = np.dot(XtWiX_XtWi, y)
    betas = np.squeeze(betas)
    return betas, XtWiX_XtWi, XtWiX_inv

def estimate_mgwr_betas(y, X, Wis, max_iter, backfitting_tol, n_jobs):
    '''
    Estimate local MGWR coefficients using backfitting.
    '''
    n = X.shape[0]
    p = X.shape[1]
    beta_global = np.linalg.lstsq(X, y, rcond=None)[0].reshape(-1)
    betas = np.tile(beta_global, (n, 1))

    def get_betas_for_j(j, betas):
        Xj = X[:, j]
        partial_residual = y.ravel() - np.sum(X * betas, axis=1)+ Xj * betas[:, j]
        new_beta_j = np.zeros(n)

        for i in range(n):
            w = np.diag(Wis[j][i])
            denominator = np.sum(w * (Xj ** 2))
            if denominator <= 1e-12:
                new_beta_j[i] = betas[i, j]
            else:
                numerator = np.sum(w * Xj * partial_residual)
                new_beta_j[i] = numerator / denominator

        return new_beta_j

    for iteration in range(max_iter):
        old_betas = betas.copy()
        betas = Parallel(n_jobs=n_jobs)(
            delayed(get_betas_for_j)(j, betas)
            for j in range(p)
        )
        betas = np.column_stack(betas)
        difference = np.max(np.abs(betas - old_betas))

        if difference < backfitting_tol:
            break

    return betas

def estimate_log_likelihood(
    y, X, Wis, W, rho, n_jobs, 
    multi=False, max_iter=None, backfitting_tol=None
):
    '''
    Estimate negative log-likelihood for rho.
    '''
    n = X.shape[0]
    I = np.identity(n)

    try:
        sign, logdet = np.linalg.slogdet(I - rho * W)
    except np.linalg.LinAlgError:
        return np.inf
    if sign <= 0 or not np.isfinite(logdet):
        return np.inf

    y_star = get_y_star(y, W, rho)
    
    if multi:
        betas = estimate_mgwr_betas(y, X, Wis, max_iter, backfitting_tol, n_jobs)
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(estimate_gwr_betas)(y_star, X, Wi)
            for Wi in Wis
        )
        betas, _, _ = zip(*results)
        betas = np.asarray(betas)

    y_hat = np.array([np.dot(X[i], betas[i]) for i in range(n)]).reshape(-1, 1)
    residuals = y_star - y_hat
    rss = np.dot(residuals.T, residuals)
    sigma2 = float(rss / n)

    if rss <= 0 or not np.isfinite(rss):
        return np.inf

    ll = float(
        logdet
        - (n / 2.0)
        * (np.log(2.0 * np.pi * sigma2) + 1.0)
    )
    return -ll

def estimate_rho(
    y, X, Wis, W, n_jobs, rho_tolerance=1.e-6, 
    multi=False, max_iter=None, backfitting_tol=None
):
    '''
    Estimate the spatial autoregressive parameter rho.
    '''
    rho_bounds = (-1 + rho_tolerance, 1 - rho_tolerance)
    rho_obj = lambda rho_: estimate_log_likelihood(
        y, X, Wis, W, rho_, n_jobs, multi, max_iter, backfitting_tol
    )
    result = minimize_scalar(rho_obj, bounds=rho_bounds, method='bounded')
    rho = float(result.x)
    return rho