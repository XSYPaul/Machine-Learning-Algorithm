import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors


class PCA:

    # calculate covariance matrix
    def calculate_cov(self, X, Y=None):
        m = X.shape[0]  # get the number of sample

        # normalization
        X = X - np.mean(X, axis=0)  # subtract mean of each feature
        Y = X if Y == None else Y - np.mean(Y, axis=0)

        return 1 / m * np.matmul(X.T, Y)

    # n dimension to n_component dimension
    def transform(self, X, n_components):
        cov_matrix = self.calculate_cov(X)  # compute covariance matrix
        eig_val, eig_vec = np.linalg.eig(cov_matrix)  # calculate eigenvalues and eigenvectors

        # sort eigenvalue and choose correspond eigenvectors
        index = eig_val.argsort()[::-1]
        eig_vec = eig_vec[:, index]
        new_features = eig_vec[:, :n_components]

        return np.matmul(X, new_features)  # reduce dimension


def main():
    '''
    Demo of how to reduce the dimensionality of the data to two dimension
    and plot the result
    '''

    data = datasets.load_digits()
    X = data.data
    y = data.target

    # project the data onto the 2 primary principal components
    X_trans = PCA().transform(X, 2)
    x1 = X_trans[:, 0]
    x2 = X_trans[:, 1]

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

    class_distr = []

    # plot the different class distributions
    for i, l in enumerate(np.unique(y)):
        _x1 = x1[y == l]
        _x2 = x2[y == l]
        _y = y[y == l]
        class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))

    plt.legend(class_distr, y, loc=1)

    plt.suptitle("PCA Dimesionality Reduction")
    plt.title("Digit Dataset")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

if __name__ == '__main__':
    main()
