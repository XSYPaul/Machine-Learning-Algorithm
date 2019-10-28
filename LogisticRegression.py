import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class LogisticRegressionClassifier:

    def __init__(self, max_iter=200, learning_rate=0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        res = 1 / (1 + np.exp(-x))
        return res

    def data_matrix(self, x):  # append x0 , bias
        data_mat = []
        for d in x:
            data_mat.append([1, *d])

        return data_mat

    def fit(self, X, y):
        data_mat = self.data_matrix(X)  # append bias
        self.weights = np.zeros((len(data_mat[0]), 1), dtype=np.float32)  # weights n * 1

        # Stochastic Gradiant Descent
        for iter_ in range(self.max_iter):
            for i in range(len(X)):
                result = self.sigmoid(np.dot(data_mat[i], self.weights))  # 1 * n   n * 1
                error = y[i] - result
                self.weights += self.learning_rate * error * np.transpose([data_mat[i]])

        print('LogisticRegression Model(learning_rate={},max_iter={})'.format(self.learning_rate, self.max_iter))

    def score(self, X_test, y_test):
        correct = 0
        X_test = self.data_matrix(X_test)
        for x, y in zip(X_test, y_test):
            result = np.dot(x, self.weights)
            if (result > 0 and y == 1) or (result < 0 and y == 0):
                correct += 1

        return correct / len(X_test)




def main():
    '''
    use iris data to test this algorithm and plot the result
    '''

    # data preparation
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    X = data[:, :2]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # build the model
    lr_clf = LogisticRegressionClassifier()
    lr_clf.fit(X_train, y_train)
    score = lr_clf.score(X_test, y_test)
    print(score)


    # plot the result
    x_points = np.arange(4, 8)
    y_ = -(lr_clf.weights[1] * x_points + lr_clf.weights[0]) / lr_clf.weights[2]
    plt.plot(x_points, y_)

    plt.scatter(X[:50, 0], X[:50, 1], label='0')
    plt.scatter(X[50:, 0], X[50:, 1], label='1')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
