## PCA is a type of dimension reduction.
import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, x):
        # calculate mean
        self.mean = np.mean(x, axis=0)
        x= x-self.mean
        # calculate covariance
        covar = np.cov(x.T)
        # print(covar.shape)
        # calculate eigenvectors, eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(covar)
        eigenvectors = eigenvectors.T
        # sort eigenvectors
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[idx]
        # store first n eigenvectors
        self.components = eigenvectors[0:self.n_components]

    def transform(self, x):
        x = x - self.mean
        return np.dot(x, self.components.T)

