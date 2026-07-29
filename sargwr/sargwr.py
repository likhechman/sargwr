__author__ = 'Alexandra Kaluzhak'

import numpy as np

from copy import deepcopy

from scipy.stats import t, norm, chi2
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar

from tqdm import tqdm
from joblib import Parallel, delayed

from .estimators import *
from .kernels import *

import warnings 
warnings.filterwarnings('ignore')

class GWR_SL:
    def __init__(
        self, kernel='gaussian', family='gaussian', selector='golden', 
        selector_metric='AICc', n_jobs=1, fixed=False, use_coords=True, spherical=False, 
        intercept=True, verbose=False, compute_deep_statistics=True
    ):
        '''
        Initialize a Geographically Weighted Regression Spatial Lag model.

        Parameters
        ----------
        kernel : str, default='gaussian'
            Kernel function used to compute geographical weights. Supported values are
            'gaussian', 'bisquare', and 'exponential'.

        family : str, default='gaussian'
            Probability distribution of the response variable. Currently only
            'gaussian' is supported.

        selector : str, default='golden'
            Bandwidth selection method. Supported values are 'golden' for Golden
            section search and 'interval' for interval search.

        selector_metric : str, default='AICc'
            Model selection criterion used for bandwidth optimization. Supported values
            are 'AIC', 'AICc', and 'BIC'.

        n_jobs : int, default=1
            Number of parallel jobs used during local coefficient estimation. A value
            of -1 uses all available CPU cores.

        fixed : bool, default=False
            If True, use a fixed (distance-based) bandwidth. Otherwise, use an adaptive
            bandwidth defined by the number of nearest neighbours.

        use_coords : bool, default=True
            If True, construct the spatial weights matrix from the supplied geographic
            coordinates. Otherwise, use the user-provided spatial weights matrix `W`.

        spherical : bool, default=False
            If True, compute great-circle distances for geographic coordinates.

        intercept : bool, default=True
            If True, automatically add an intercept column to the design matrix.

        verbose : bool, default=False
            If True, print model information during fitting.
        
        compute_deep_statistics : bool, default=True
            If True computes local and other statistics that isn't nesseccary for AIC, 
            AICc or BIC evaluation.
        '''   
        self.kernel = kernel.lower()
        self.family = family.lower()
        self.selector = selector.lower()
        self.selector_metric = selector_metric.lower()
        self.n_jobs = n_jobs
        self.fixed = fixed
        self.use_coords = use_coords
        self.spherical = spherical
        self.intercept = intercept
        self.verbose = verbose
        self.compute_deep_statistics = compute_deep_statistics

        self.distance_matrix = None

    def _add_intercept(self, X):
        '''
        Add an intercept column to the design matrix.
        '''
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if not np.allclose(X[:, 0], 1):
            X = np.column_stack((np.ones(X.shape[0]), X))

        return X

    def _bandwidth_objective(self, bandwidth, model=None):
        '''
        Evaluate the objective function used for bandwidth selection.

        The model is fitted using the supplied bandwidth, and the selected model
        selection criterion (AIC, AICc, or BIC) is returned.

        Parameters
        ----------
        bandwidth : float
            Candidate bandwidth. Adaptive bandwidths are rounded to the nearest
            integer before model fitting.

        Returns
        -------
        float
            Value of the selected model selection criterion.
        '''
        if not self.fixed:
            bandwidth = int(round(bandwidth))
        model.fit(
            self.coords, self.y, self.X, bandwidth=bandwidth, W=self.W, 
            rho_tolerance=self.rho_tolerance
        )
        if self.selector_metric == 'aic':
            return model.aic
        if self.selector_metric == 'aicc':
            return model.aicc
        if self.selector_metric == 'bic':
            return model.bic
    
    def _select_bandwidth(self):
        '''
        Select the optimal bandwidth.

        If no search interval is provided, a default interval is constructed. 
        The bandwidth is then selected using either golden section search or interval 
        search.

        Returns
        -------
        int or float
            Optimal bandwidth. For adaptive kernels the returned value is an integer
            representing the number of nearest neighbours. For fixed kernels the
            returned value is a distance.
        '''
        if self.verbose:
            print('Selecting bandwidth')
        
        model = deepcopy(self)
        model.verbose = False
        model.compute_deep_statistics = False

        if self.bandwidth_interval is None:
            if self.fixed:
                lower = 1.0e-6
                upper = max(self.distance_matrix.ravel())
            else:
                lower = 10
                upper = self.n - 1
        else:
            if len(self.bandwidth_interval) != 2:
                raise ValueError(
                    'bandwidth_interval must be None or contain exactly two values.'
                    )
            lower = self.bandwidth_interval[0]
            upper = self.bandwidth_interval[1]

        if self.selector == 'golden':
            bandwidth_objective = lambda bandwidth: self._bandwidth_objective(bandwidth, model=model)
            result = minimize_scalar(
                bandwidth_objective,
                bounds=(lower, upper),
                method='bounded',

            )
            best_bandwidth = int(round(result.x))
        elif self.selector == 'interval':
            bandwidths = range(lower, upper + 1)
            scores = Parallel(n_jobs=self.n_jobs)(
                delayed(self._bandwidth_objective)(bandwidth)
                for bandwidth in tqdm(
                    bandwidths, 
                    total=len(bandwidths), 
                    desc='searching best bandwidths: '
                )
            )
            best_bandwidth = bandwidths[np.argmin(scores)]
        else:
            raise ValueError('Unsuported selector for bandwidth')
        return best_bandwidth
    
    def _compute_local_weights(self, bandwidth):
        '''
        Compute local spatial weighting matrices.

        Construct a diagonal spatial weighting matrix for each observation according
        to the selected kernel function and bandwidth.

        Parameters
        ----------
        bandwidth : int or float
            Kernel bandwidth. For adaptive kernels this represents the number of
            nearest neighbours. For fixed kernels this represents a distance.

        Returns
        -------
        list of ndarray
            List containing one diagonal weight matrix of shape (n, n) for each
            observation.
        '''
        if self.verbose:
            print('Compute local weights')
        self.Kernel.bandwidth = bandwidth
        Wis = [self.Kernel.get_Wi(i) for i in range(self.n)]
        return Wis

    def _compute_hat_matrix(self):
        '''
        Compute the GWR hat matrix.

        The hat matrix maps the transformed response variable to its fitted values and
        is used to calculate the effective number of model parameters and related
        diagnostic statistics.

        Returns
        -------
        ndarray
            Hat matrix of shape (n, n).
        '''
        S = np.array([
            np.dot(self.X[i], self.XtWiX_XtWis[i]) for i in range(self.n)
        ])
        return S
    
    def _compute_local_stastics(self):
        '''
        Compute local t-values and p-values.

        Returns
        -------
        local_ts : ndarray
            List containing local t-values of shape (n, p) for each observation.
        local_ps : ndarray
            List containing local p-values of shape (n, p) for each observation.
        '''
        if self.verbose:
            print('Compute local stastics')
        local_ts = np.array([
            self.betas[i] / np.sqrt(np.diag(self.sigma2 * self.XtWiX_invs[i])) for i in range(self.n)
        ])
        local_ps = 2 * (1 - t.cdf(np.abs(local_ts), df=self.df))
        return local_ts, local_ps

    def _compute_lm_test(self):
        '''
        Compute Lagrange Multiplier test for spatial lag dependence.

        Returns
        -------
        float
            LM statistic.

        float
            P-value.
        '''

        y_star = self.y

        betas = np.asarray([estimate_local_betas(y_star, self.X, Wi)[0] for Wi in self.Wis])
        y_hat = np.array([np.dot(self.X[i], betas[i]) for i in range(self.n)]).reshape(-1, 1)
        e = y_star - y_hat

        sigma2 = float(np.dot(e.T, e) / self.n)
        We = np.dot(self.W, e)
        num = float(np.dot(e.T, We) ** 2)

        den = sigma2 ** 2 * np.trace(np.dot(self.W.T, self.W) + np.dot(self.W, self.W))
        LM = num / den
        p = 1.0 - chi2.cdf(LM, 1)

        return LM, p
    
    def _compute_diagnostics(self):
        '''
        Compute model diagnostic statistics.

        Calculates residual sum of squares, residual variance, Gaussian
        log-likelihood, coefficient of determination, effective degrees of freedom,
        AIC, AICc, BIC, and the Wald test for the spatial autoregressive parameter.

        Notes
        -----
        The log-likelihood is evaluated using the transformed response variable

            y* = (I - rho W) y,

        which is consistent with maximum likelihood estimation for the spatial lag
        model.
        '''
        if self.verbose:
            print('Compute diagnostics')

        self.rss = float(np.dot(self.residuals.T, self.residuals))
        self.sigma2 = float(self.rss / self.n)

        sign, logdet = np.linalg.slogdet(self.I - self.rho * self.W)
        self.loglik = float(
            logdet
            - (self.n / 2.0)
            * (np.log(2.0 * np.pi * self.sigma2) + 1.0)
        )
        self.tss = float(np.sum((self.y - np.mean(self.y))**2))
        self.S = self._compute_hat_matrix()
        self.trace_S = float(np.trace(self.S))
        self.df = self.n - self.trace_S

        self.r2 = 1.0 - self.rss / self.tss
        self.adj_r2 = 1 - (self.rss / self.df) / (self.tss / (self.n - 1))

        self.aic = -2 * self.loglik + 2 * self.trace_S
        self.aicc = (
            self.aic + (2 * self.trace_S * (self.trace_S + 1)) / (self.n - self.trace_S - 1)
        )
        self.bic = -2 * self.loglik + np.log(self.n) * self.trace_S

        if self.compute_deep_statistics:
            ll0 = estimate_log_likelihood(self.y, self.X, self.Wis, self.W, self.rho, self.n_jobs)
            ll1 = estimate_log_likelihood(self.y, self.X, self.Wis, self.W, self.rho + self.rho_tolerance, self.n_jobs)
            ll2 = estimate_log_likelihood(self.y, self.X, self.Wis, self.W, self.rho - self.rho_tolerance, self.n_jobs)
            second = (ll1 - 2 * ll0 + ll2) / self.rho_tolerance**2

            if second <= 0 or not np.isfinite(second):
                self.se_rho = np.nan
                self.wald_z = np.nan
                self.wald_p_rho = np.nan
            else:
                self.se_rho = np.sqrt(1 / second)
                self.wald_z = self.rho / self.se_rho
                self.wald_p_rho = 2 * (1 - norm.cdf(abs(self.wald_z)))

            ll_full = self.loglik
            ll_restricted = -estimate_log_likelihood(self.y, self.X, self.Wis, self.W, 0, self.n_jobs)
            self.lr = 2 * (ll_full - ll_restricted)
            self.lr_pvalue = 1.0 - chi2.cdf(self.lr, df=1)

            self.lm, self.lm_pvalue = self._compute_lm_test()

            alphas = np.array([.1, .05, .001])
            self.alpha_adj = (alphas * self.p) / self.trace_S

            if self.alpha is None:
                self.alpha = np.abs(self.alpha_adj[1]) / 2.0
                self.critical = t.ppf(1 - self.alpha, self.n - 1)
            else:
                self.alpha = np.abs(self.alpha) / 2.0
                self.critical = t.ppf(1 - self.alpha, self.n - 1)
            
            self.leverage = np.diag(self.S)
            self.studentized_residuals = self.residuals.ravel() / np.sqrt(
                self.sigma2 * np.maximum(1.0 - self.leverage, 1e-12)
            )
            self.cooks_distance = (
                self.studentized_residuals**2
                * np.maximum(self.leverage, 1e-12)
                / (self.p * (1.0 - np.maximum(self.leverage, 1e-12)))
            )

    def fit(
        self, coords, y, X, W=None, bandwidth=None, variable_names=None, 
        bandwidth_interval=None, alpha=None, rho_tolerance=1.e-5
    ):
        '''
        Fit the Geographically Weighted Regression model Spatial Lag model.

        Parameters
        ----------
        coords : ndarray of shape (n, 2)
            Geographic coordinates of the observations.

        y : ndarray of shape (n,)
            Response variable.

        X : ndarray of shape (n, p)
            Matrix of explanatory variables. If `intercept=True`, a column of ones is
            automatically added.

        W : ndarray or libpysal.weights.W, optional
            Spatial weights matrix used in the spatial lag model. Required when
            `use_coords=False`.

        bandwidth : int or float, optional
            Kernel bandwidth. If None, the bandwidth is selected automatically.

        variable_names : list of str, optional
            Names of the explanatory variables used in the model summary.

        bandwidth_interval : tuple, list, or None, optional
            Lower and upper bounds for bandwidth selection. Must contain exactly two
            values.

        alpha : int or float, optional
            critical value to determine which t-values are associated with statistically 
            significant parameter estimates. Default to None in which case the adjusted
            alpha value at the 95 percent CI is automatically used.

        rho_tolerance : float, default=1e-5
            Numerical tolerance used when optimizing the spatial autoregressive
            parameter.

        Returns
        -------
        None

        Notes
        -----
        Model estimation proceeds as follows:

        1. Construct local geographical weighting matrices.
        2. Estimate the spatial autoregressive parameter rho by maximum likelihood.
        3. Transform the response variable using y* = (I - rho W)y.
        4. Estimate local regression coefficients using weighted least squares.
        5. Compute fitted values and model diagnostics.
        '''

        self.coords = np.asarray(coords)
        self.y = np.asarray(y)
        self.X = np.asarray(X)

        if (len(self.coords) != len(self.y)) or (len(self.coords) != len(self.X)):
            raise ValueError('coords must be the same length as X and y')
        if len(self.X) != len(self.y):
            raise ValueError('X and y must be the same length')
        if self.coords.shape[1] != 2:
            raise ValueError('coords must have exactly two coordinates')

        self.rho_tolerance = rho_tolerance
        self.alpha = alpha
        self.W = W
        self.bandwidth_interval = bandwidth_interval
        
        if self.distance_matrix is None:
            if self.verbose:
                print('Compute distance matrix')
            self.Kernel = Kernel(
                self.coords, 1, function=self.kernel, 
                fixed=self.fixed, spherical=self.spherical
            )
            self.distance_matrix = self.Kernel.compute_distance_matrix()

        if self.intercept:
            self.X = self._add_intercept(self.X)

        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.I = np.identity(self.n)

        if bandwidth is None:
            self.bandwidth = self._select_bandwidth()
        else:
            self.bandwidth = bandwidth

        if variable_names is None:
            self.variable_names = [f'X{i}' for i in range(self.p)]
        else:
            self.variable_names = variable_names
            if self.intercept:
                self.variable_names.insert(0, 'Intercept')

        self.Wis = self._compute_local_weights(self.bandwidth)

        if self.use_coords:
            self.W = np.array([np.diag(Wi) for Wi in self.Wis])
        else:
            if self.W is None:
                raise ValueError('Provide W or set use_coords to True')
            else:
                if hasattr(W, 'full'):
                    self.W = W.full()[0]
                else:
                    self.W = np.asarray(W)

        if (len(self.W) != len(self.y)) or (len(self.W) != len(self.X)):
            raise ValueError('W must be the same length as X and y')

        if self.family == 'gaussian':
            if self.verbose:
                print('Estimating rho')
            self.rho = estimate_rho(
                self.y, self.X, self.Wis, self.W, self.n_jobs, 
                rho_tolerance=self.rho_tolerance
            )
            self.y_star = get_y_star(self.y, self.W, self.rho)

            if self.verbose:
                print('Estimating local betas')
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(estimate_local_betas)(self.y_star, self.X, Wi)
                for Wi in self.Wis
            )
            betas, XtWiX_XtWis, XtWiX_invs = zip(*results)
            self.betas = np.asarray(betas)
            self.XtWiX_XtWis = np.asarray(XtWiX_XtWis)
            self.XtWiX_invs = np.asarray(XtWiX_invs)
            
            self.y_hat = np.array([
                np.dot(self.X[i], self.betas[i]) for i in range(self.n)
            ]).reshape(-1, 1)
            self.residuals = self.y_star - self.y_hat

            self._compute_diagnostics()
            if self.compute_deep_statistics:
                self.local_ts, self.local_ps = self._compute_local_stastics()
        else:
            raise ValueError('Unsupported family type', self.family)

    def filter_tvalues(self, alpha=None, critical_t=None):
        '''
        Identify statistically significant local regression coefficients.

        Compare the absolute local t-values against a critical t-value and return
        a boolean array indicating whether each local coefficient is statistically
        significant.

        Parameters
        ----------
        alpha : float, optional
            Significance level used to compute the critical t-value. If ``None``,
            the model's adjusted significance level (``self.alpha_adj[1]``) is used.

        critical_t : float, optional
            User-specified critical t-value. If provided, this value is used
            directly and ``alpha`` is ignored.

        Returns
        -------
        ndarray of bool, shape (n, p)
            Boolean array where ``True`` indicates that the corresponding local
            coefficient is statistically significant and ``False`` otherwise.

        Raises
        ------
        ValueError
            If local t-values have not been computed.
        '''

        if not hasattr(self, 'local_ts'):
            raise ValueError(
                'Local t-values have not been computed.'
            )

        if critical_t is None:
            if alpha is None:
                alpha = self.alpha_adj[1]
            critical_t = t.ppf(1.0 - alpha / 2.0, df=self.df)

        significant = np.abs(self.local_ts) >= critical_t

        return significant

    def compute_local_collinearity(self):
        '''
        Compute local multicollinearity diagnostics.

        Computes, for each regression location:

        - local correlation coefficients,
        - local variance inflation factors (VIF),
        - local condition numbers,
        - local variance-decomposition proportions.

        Returns
        -------
        local_corr : ndarray, shape (n, p, p)
            Local correlation matrices.

        local_vif : ndarray, shape (n, p)
            Local variance inflation factors.

        local_cn : ndarray, shape (n,)
            Local condition numbers.

        local_vdp : ndarray, shape (n, p, p)
            Local variance-decomposition proportions.

        Raises
        ------
        ValueError
            If model has not been fitted yet.
        '''

        if not hasattr(self, 'Wis'):
            raise ValueError('Model has not been fitted yet.')

        local_corr = np.zeros((self.n, self.p, self.p))
        local_vif = np.zeros((self.n, self.p))
        local_cn = np.zeros(self.n)
        local_vdp = np.zeros((self.n, self.p, self.p))

        for i in range(self.n):
            w = np.diag(self.Wis[i])
            Xw = self.X * np.sqrt(w)[:, None]

            corr = np.corrcoef(Xw, rowvar=False)
            local_corr[i] = corr

            for j in range(self.p):
                y = Xw[:, j]
                X = np.delete(Xw, j, axis=1)
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                yhat = np.dot(X, beta)
                rss = np.sum((y - yhat) ** 2)
                tss = np.sum((y - y.mean()) ** 2)
                r2 = 1 - rss / tss if tss > 0 else 0
                local_vif[i, j] = 1 / (1 - r2) if r2 < 1 else np.inf

           
            XtWX = np.dot(Xw.T, Xw)
            eigvals, eigvecs = eigh(XtWX)
            eigvals = np.maximum(eigvals, 1e-12)
            local_cn[i] = np.sqrt(eigvals.max() / eigvals.min())

            phi = (eigvecs ** 2) / eigvals
            local_vdp[i] = phi / phi.sum(axis=1, keepdims=True)

        self.local_corr = local_corr
        self.local_vif = local_vif
        self.local_cn = local_cn
        self.local_vdp = local_vdp

        return local_corr, local_vif, local_cn, local_vdp

    def monte_carlo_test(self, n_perm=100, random_state=None):
        '''
        Monte Carlo test for spatial variability of local coefficients.

        Parameters
        ----------
        n_perm : int, default=99
            Number of random permutations.

        random_state : int, optional
            Seed for the random number generator.

        Returns
        -------
        pvalues : ndarray, shape (p,)
            Monte Carlo p-values for each coefficient surface.

        observed : ndarray, shape (p,)
            Observed variances of the local coefficients.

        simulated : ndarray, shape (n_perm, p)
            Simulated variances from the permutations.
        
        Raises
        ------
        ValueError
            If model has not been fitted yet.
        '''

        rng = np.random.default_rng(random_state)

        if not hasattr(self, 'betas'):
            raise ValueError('Model has not been fitted yet.')

        observed = np.std(self.betas, axis=0)
        simulated = np.zeros((n_perm, self.p))
        verbose = self.verbose
        self.verbose = False

        for r in tqdm(range(n_perm), total=n_perm, desc='permutations: '):
            coords_perm = self.coords[rng.permutation(self.n)]
            model = deepcopy(self)
            model.distance_matrix = None 

            model.fit(
                coords_perm,
                self.y,
                self.X,
                bandwidth=self.bandwidth,
                W=self.W,
                rho_tolerance=self.rho_tolerance
            )

            simulated[r] = np.std(model.betas, axis=0)

        pvalues = (np.sum(simulated >= observed, axis=0)) / float(n_perm)

        self.verbose = verbose
        self.mc_observed = observed
        self.mc_simulated = simulated
        self.mc_pvalues = pvalues

        return pvalues
    
    def predict(self, coords_new=None, X_new=None):
        '''
        Predict fitted values from a fitted SARGWR model.

        Parameters
        ----------
        coords_new : ndarray of shape (m, 2), optional
            Coordinates for new prediction locations. If None, in-sample fitted
            values are returned.

        X_new : ndarray of shape (m, p), optional
            Design matrix for new observations. Required when coords_new is not None.

        Returns
        -------
        ndarray
            Predicted values. Shape is (n,) for in-sample prediction and (m,) for
            out-of-sample prediction.

        Raises
        ------
        ValueError
            If model has not been fitted yet.

        ValueError
            If coords_new or X_new was not provided.

        ValueError
            If coords_new and X_new have different shapes.
        '''

        if not hasattr(self, 'betas'):
            raise ValueError('Model has not been fitted yet.')

        if coords_new is None and X_new is None:
            return self.y_hat.ravel()

        if coords_new is None or X_new is None:
            raise ValueError('coords_new and X_new must both be provided for prediction.')

        coords_new = np.asarray(coords_new)
        X_new = np.asarray(X_new)

        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, self.X.shape[1])

        if coords_new.shape[0] != X_new.shape[0]:
            raise ValueError('coords_new and X_new must have the same number of rows.')

        if self.intercept:
            X_new = self._add_intercept(X_new)

        m = coords_new.shape[0]
        preds = np.zeros(m)

        K_predict = Kernel(
            self.coords, self.bandwidth, function=self.kernel, fixed=self.fixed, spherical=self.spherical
        )
        K_predict.compute_distance_matrix(coords_new=coords_new)

        preds = [
            np.dot(
                X_new[i], 
                estimate_local_betas(self.y_star, self.X, K_predict.get_Wi(i))[0]
            ) for i in range(m)
        ] 
        preds = np.asarray(preds)

        return preds

    def summary(self):
        '''
        Return a formatted summary of the fitted GWR-SL model.

        Returns
        -------
        str
            Model summary.
        '''

        lines = []

        def section(title):
            lines.append('')
            lines.append(title)
            lines.append('-' * 104)

        def two_col(left_label, left_value, right_label='', right_value=''):
            lines.append(
                f'{left_label:<34}{left_value:>16}'
                f'    '
                f'{right_label:<34}{right_value:>16}'
            )

        lines.append('=' * 104)
        lines.append('Geographically Weighted Regression - Spatial Lag model')
        lines.append('=' * 104)

        section('Model Information')

        bw = f'{"Fixed" if self.fixed else "Adaptive"} {self.bandwidth}'

        two_col('Number of observations', self.n,
                'Number of predictors', self.p)

        two_col('Kernel', self.kernel,
                'Bandwidth', bw)

        section('Model Diagnostics')

        two_col('Residual sum of squares', f'{self.rss:.6f}',
                'Sigma²', f'{self.sigma2:.6f}')

        two_col('Log-likelihood', f'{self.loglik:.6f}',
                'AIC', f'{self.aic:.6f}')

        two_col('AICc', f'{self.aicc:.6f}',
                'BIC', f'{self.bic:.6f}')

        two_col('R²', f'{self.r2:.6f}',
                'Adjusted R²', f'{self.adj_r2:.6f}')

        two_col('trace(S)', f'{self.trace_S:.6f}',
                'Degrees of freedom', f'{self.df:.6f}')

        two_col('Adjusted α (95%)',
                f'{self.alpha_adj[1]:.6f}')

        section('Residual Diagnostics')

        two_col(
            'Mean residual',
            f'{np.mean(self.residuals):.6f}',
            'Std. residual',
            f'{np.std(self.residuals, ddof=1):.6f}'
        )

        two_col(
            'Mean leverage',
            f'{np.mean(self.leverage):.6f}',
            'Maximum leverage',
            f'{np.max(self.leverage):.6f}'
        )

        two_col(
            'Mean Cook\'s distance',
            f'{np.mean(self.cooks_distance):.6f}',
            'Maximum Cook\'s distance',
            f'{np.max(self.cooks_distance):.6f}'
        )

        two_col(
            'Mean |studentized residual|',
            f'{np.mean(np.abs(self.studentized_residuals)):.6f}',
            'Maximum |studentized residual|',
            f'{np.max(np.abs(self.studentized_residuals)):.6f}'
        )

        section('Summary Statistics for Local Coefficients')

        lines.append(
            f'{"Variable":<29}'
            f'{"Mean":>15}'
            f'{"Std":>15}'
            f'{"Min":>15}'
            f'{"Median":>15}'
            f'{"Max":>15}'
        )

        lines.append('-' * 104)

        for j, name in enumerate(self.variable_names):

            beta = self.betas[:, j]

            lines.append(
                f'{name:<29}'
                f'{np.mean(beta):15.6f}'
                f'{np.std(beta, ddof=1):15.6f}'
                f'{np.min(beta):15.6f}'
                f'{np.median(beta):15.6f}'
                f'{np.max(beta):15.6f}'
            )

        section('Statistics for autoregressive coefficient ρ')

        two_col('Value', f'{self.rho:.6f}',
                'Standard error', f'{self.se_rho:.6f}')
        
        two_col('Wald test z-statistic', f'{self.wald_z:.6f}',
                'Wald test p-value', f'{self.wald_p_rho:.6f}')
        
        two_col('Likelihood ratio', f'{self.lr:.6f}',
                'Likelihood ratio p-value', f'{self.lr_pvalue:.6f}')

        two_col('Lagrange Multiplier statistic', f'{self.lm:.6f}',
                'Lagrange Multiplier p-value', f'{self.lm_pvalue:.6f}')

        return '\n'.join(lines)