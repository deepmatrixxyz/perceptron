import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv('./penguins.csv')

X = dataset.iloc[:, 1:].to_numpy()
y = dataset.species.to_numpy()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.1, random_state=10)


X_train, X_validation, y_train, y_validation = train_test_split(
    X_train, y_train, test_size = 0.15, random_state=10)


scaler = StandardScaler().fit(X_train)


X_train = scaler.transform(X_train)
X_validation = scaler.transform(X_validation)
X_test = scaler.transform(X_test)


class Perceptron:
    def __init__(self, X_matrix, y_matrix, epochs=1, n_features=6, batch_size=1):
        self.X = X_matrix
        self.y = y_matrix
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = np.random.randn(n_features)
        self.error = np.random.randn()

    def propagation(self, weights, x_matrix):
        y_hat = np.matmul(weights, x_matrix.transpose()) + self.error
        return y_hat

    def loss_function(self, y_true, y_hat):
        boundary = y_true * y_hat
        boundary = np.where(boundary > 1, 0, 1 - boundary)
        return boundary.mean()

    def derivatives(self, y_true, y_hat, x_matrix):
        boundary = y_true * y_hat
        reference = np.where(boundary > 1, 0, -y_true)
        weights_dvt = (np.expand_dims(reference, 1) * x_matrix).mean(0)
        errors_dvt = reference.mean()
        return weights_dvt, errors_dvt

    def weights_correction(self, weights_dvt, errors_dvt):
        self.weights -= weights_dvt
        self.error -= errors_dvt

    def train(self):
        for epoch in range(self.epochs):
            for index in np.arange(0, len(self.y), self.batch_size):
                x_instance, y_instance = (self.X[index: index + self.batch_size], \
                                         self.y[index: index + self.batch_size])
                y_hat = self.propagation(self.weights, x_instance)
                loss_value = self.loss_function(y_instance, y_hat)
                weights_dvt, errors_dvt = self.derivatives(y_instance, y_hat, \
                                                           x_instance)
                self.weights_correction(weights_dvt, errors_dvt)

            #if loop % 1 == 0:
                y_hat_validation = self.propagation(self.weights, X_validation)
                loss_value_validation = self.loss_function(y_validation, y_hat_validation)
                print(f'epoch {epoch} and batch {index}-{index + self.batch_size} ===> {loss_value_validation}')




model = Perceptron(X_train, y_train, n_features=6, batch_size=2)
model.train()


y_hat = model.propagation(model.weights, X_test)
y_hat = np.where(y_hat > 0, 1, -1)
(y_hat == y_test).mean()










