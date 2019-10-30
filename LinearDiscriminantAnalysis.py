import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification


class LDA():
    def Train(self, X, y):
        X1 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
        X2 = np.array([X[i] for i in range(len(X)) if y[i] == 1])

        # calculate mean
        mju1 = np.mean(X1, axis=0)
        mju2 = np.mean(X2, axis=0)

        cov1 = np.dot((X1 - mju1).T, (X1 - mju1))
        cov2 = np.dot((X2 - mju2).T, (X2 - mju2))
        Sw = cov1 + cov2

        # calculate w
        w = np.dot(np.mat(Sw).I, (mju1 - mju2).reshape((len(mju1), 1)))

        # record training result
        self.mju1 = mju1  # the mean of class_1
        self.cov1 = cov1
        self.mju2 = mju2  # the mean of class_2
        self.cov2 = cov2
        self.Sw = Sw  # 类内散度矩阵
        self.w = w  # 判别权重矩阵

    def Test(self, X, y):

        y_new = np.dot((X), self.w)

        # 计算fisher线性判别式
        nums = len(y)
        c1 = np.dot((self.mju1 - self.mju2).reshape(1, (len(self.mju1))), np.mat(self.Sw).I)
        c2 = np.dot(c1, (self.mju1 + self.mju2).reshape((len(self.mju1), 1)))
        c = 1 / 2 * c2
        h = y_new - c

        # 判别
        y_hat = []
        for i in range(nums):
            if h[i] >= 0:
                y_hat.append(0)
            else:
                y_hat.append(1)

        # 计算分类精度
        count = 0
        for i in range(nums):
            if y_hat[i] == y[i]:
                count += 1
        precise = count / nums

        print("Numbers of test samples:", nums)
        print("Numbers of predict correct sample:", count)
        print("Test precise:", precise)

        return precise

if __name__ == '__main__':
    n_samples = 500
    X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                               n_classes=2, n_informative=1, n_clusters_per_class=1, class_sep=0.5, random_state=10)

    # LDA binary classification
    lda = LDA()
    Xtrain = X[:299, :]
    Ytrain = y[:299]
    Xtest = X[300:, :]
    Ytest = y[300:]
    lda.Train(Xtrain, Ytrain)
    precise = lda.Test(Xtest, Ytest)

    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Test precise:" + str(precise))
    plt.show()

