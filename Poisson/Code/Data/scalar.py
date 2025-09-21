import numpy as np
import scipy.spatial.distance as distance
import scipy.stats.qmc as qmc
import copy
import math

class ScalarGenerator1D:
    def __init__(self, size=1.0, nCell=32, nKnot=4):
        """
        1D Gaussian Process scalar generator.
        :param size: length of domain
        :param nCell: number of grid cells
        :param nKnot: number of GP knots
        """
        self.size  = size
        self.nCell = nCell
        self.nKnot = nKnot
        self.lenScale = 1.0  # GP length scale

        # Knots for GP
        self.xKnot = np.linspace(0, size, nKnot).reshape(-1,1)  # (nKnot,1)

        # Squared distance matrix between knots
        knotDistMat = distance.cdist(self.xKnot, self.xKnot, 'sqeuclidean')
        # Knot covariance matrix
        knotCovMat = np.exp(-knotDistMat / self.lenScale)
        self.knotCovMatInv = np.linalg.inv(knotCovMat)

        # Grid cell coordinates (cell-centered)
        h = size / nCell
        self.xGrid = np.array([(i+0.5)*h for i in range(nCell)]).reshape(-1,1)

        # Covariance matrix between grid cells and knots
        self.covMat = distance.cdist(self.xGrid, self.xKnot, 'sqeuclidean')
        self.covMat = np.exp(-self.covMat / self.lenScale)

    def generate_scalar1d(self, nSample, valMin=0.0, valMax=1.0, strictMin=False):
        """
        Generate nSample 1D scalar fields
        """
        # Sobol sequence for quasi-random knot values
        pow = int(np.log2(self.nKnot*nSample)) + 1
        sobolSeq = qmc.Sobol(d=1).random_base2(m=pow)
        sobolSeq = sobolSeq * (valMax - valMin) + valMin
        np.random.shuffle(sobolSeq)

        samples = np.zeros((nSample, self.nCell))

        # Interpolate each sample using GP
        s, e = 0, 0
        for i in range(nSample):
            s, e = e, e + self.nKnot
            knots = copy.deepcopy(sobolSeq[s:e])
            # GP interpolation: f_grid = C_grid,knot * C_knot^-1 * f_knots
            sca = np.matmul(self.covMat, np.dot(self.knotCovMatInv, knots))
            samples[i,:] = np.squeeze(sca)

        if strictMin:
            scalarMin = np.min(samples)
            samples = samples - scalarMin + valMin

        return samples
