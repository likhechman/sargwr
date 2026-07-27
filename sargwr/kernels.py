__author__ = 'Alexandra Kaluzhak'

import numpy as np
from scipy.spatial.distance import cdist

R = 6371.0088

class Kernel:
    def __init__(
        self, coords, bandwidth, distance_matrix=None, function='gaussian', fixed=False, 
        spherical=False, eps=1.0000001
    ):
        '''
        Initialize a kernel weighting object.

        Parameters
        ----------
        coords : ndarray of shape (n, 2)
            Geographic coordinates of the observations.

        bandwidth : int or float
            Kernel bandwidth. For adaptive kernels this represents the number of
            nearest neighbours. For fixed kernels this represents a distance.
        
        distance_matrix : ndarray of shape (n, n)
            Pairwise distances matrix between observations

        function : str, default='gaussian'
            Kernel function used to transform distances into weights. Supported values
            are 'gaussian', 'bisquare', 'triangular', 'uniform', 'quadratic',
            'quartic', and 'exponential'.

        fixed : bool, default=False
            If True, use a fixed (distance-based) bandwidth. Otherwise, use an adaptive
            bandwidth determined by the specified number of nearest neighbours.

        spherical : bool, default=False
            If True, great-circle distances are used instead of Euclidean distances.

        eps : float, default=1.0000001
            Small inflation factor applied to adaptive bandwidths to ensure the
            boundary neighbour receives a non-zero weight.

        Raises
        ------
        ValueError
            If the supplied bandwidth is not positive.
        '''
        self.coords = coords
        self.bandwidth = bandwidth
        self.distance_matrix = distance_matrix
        self.function = function
        self.fixed = fixed
        self.spherical = spherical
        self.eps = eps

        if self.bandwidth <= 0:
            raise ValueError('bandwidth must be positive')

    def _kernel_weights(self, d):
        '''
        Convert distances into kernel weights.

        The supplied distances are scaled by the kernel bandwidth and transformed into
        weights according to the selected kernel function.

        Parameters
        ----------
        d : ndarray
            Distances between the target observation and all observations.

        Returns
        -------
        ndarray
            Kernel weights corresponding to the supplied distances.

        Raises
        ------
        ValueError
            If an unsupported kernel function is specified.
        '''

        u = d / self.h

        if self.function == 'triangular':
            return 1 - u
        elif self.function == 'bisquare':
            return np.where(u < 1.0, (1.0 - (u**2)) ** 2, 0.0)
        elif self.function == 'gaussian':
            return np.exp(-0.5 * (u**2))
        elif self.function == 'exponential':
            return np.exp(-u)
        elif self.function == 'uniform':
            return np.full(u.shape, 0.5)
        elif self.function == 'quadratic':
            return (3. / 4) * (1 - (u**2))
        elif self.function == 'quartic':
            return (15. / 16) * (1 - (u**2))**2
        else:
            raise ValueError(f'Unsupported kernel function: {self.function}')

    def get_Wi(self, i):
        '''
        Compute the local spatial weighting matrix.

        Construct the diagonal kernel weighting matrix associated with a single
        observation. The bandwidth is interpreted as either a fixed distance or an
        adaptive number of nearest neighbours depending on the value of `fixed`.

        Parameters
        ----------
        i : int
            Index of the target observation.

        Returns
        -------
        ndarray
            Diagonal spatial weighting matrix of shape (n, n).
        '''

        if self.distance_matrix is None:
            self.compute_distance_matrix()
        d = self.distance_matrix[i]

        if self.fixed:
            self.h = float(self.bandwidth)
        else:
            self.h = np.partition(
                d.ravel(), int(self.bandwidth) - 1
            )[int(self.bandwidth) - 1] * self.eps

        w = self._kernel_weights(d)
        w = np.diag(w)
        return w

    def compute_distance_matrix(self, coords_new=None):
        '''
        Compute pairwise distances between observations.

        Returns
        -------
        ndarray
            Euclidean or spherical distances between any pair of observation coordinates.
        '''
        if self.spherical:
            if coords_new is None:
                lat = np.radians(self.coords[:, 0])
                lon = np.radians(self.coords[:, 1])

                lat1 = lat[:, None]
                lat2 = lat[None, :]
                lon1 = lon[:, None]
                lon2 = lon[None, :]
            else:
                lat_new = np.radians(coords_new[:, 0])
                lon_new = np.radians(coords_new[:, 1])
                lat = np.radians(self.coords[:, 0])
                lon = np.radians(self.coords[:, 1])

                lat1 = lat_new[:, None]
                lat2 = lat[None, :]
                lon1 = lon_new[:, None]
                lon2 = lon[None, :]

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = (
                np.sin(dlat / 2.0) ** 2
                + np.cos(lat1) * np.cos(lat2)
                * np.sin(dlon / 2.0) ** 2
            )

            distance_matrix = 2 * R * np.arcsin(np.sqrt(a))
        else:
            if coords_new is None:
                distance_matrix = cdist(self.coords, self.coords)
            else:
                distance_matrix = cdist(coords_new, self.coords)

        self.distance_matrix = distance_matrix
        return distance_matrix