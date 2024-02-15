from dataclasses import dataclass
from simple_parsing.helpers import Serializable
import numpy as np
from sklearn.metrics import accuracy_score

@dataclass
class ModelArgs(Serializable):
    n_features: int
    n_classes: int
    lr: float = 0.001

class LogisticRegressionModel(object):
    
    def __init__(self, args: ModelArgs):
        self.args = args
        self.n_features = args.n_features
        self.n_classes = args.n_classes
        self.lr = args.lr

        # Initialize weights and bias
        self.w = np.random.rand(self.n_features + 1,1) # +1 because of intercept
        self.b = 0

    def train(self, x, y, epochs = 10, validation_dataset = None):

        x = self.add_intercept(x)

        for epoch in range(1, epochs + 1):
            # Calculate z (w1*x1+w2*x2+...+wn*xn + b)
            z = np.dot(x, self.w)

            # Compute probabilities
            h = self.__sigmoid(z)

            print('Z', z)
            print('H', h)

            # Calculate loss
            l = self.loss(y, h)
            
            # Calculate gradient
            dw, db = self.gradient(x, y, h)

            # Update weights
            self.w -= self.lr*dw
            self.b -= self.lr*db

            # Calculate accuracy
            y_hat = self.predict(x)
            acc = accuracy_score(y, y_hat)

            # If validation dataset provided, calculate validation accuracy
            if validation_dataset:
                (X_test, y_test) = validation_dataset
                X_test = self.add_intercept(X_test)
                y_hat_test = self.predict(X_test)
                val_acc = accuracy_score(y_test, y_hat_test)

            # Print
            print(f"Epoch {epoch}, acc: {acc:.2f}, loss: {l:.4f}", end = "")
            if validation_dataset:
                print(f", val_acc: {val_acc:.2f}", end = "")
            
            print("")
    
    def add_intercept(self, x):
        intercept = np.ones((x.shape[0], 1))
        return np.concatenate((intercept, x), axis = 1)

    def loss(self, y, h):
        N = y.size
        print(y.shape, np.log(h).shape, (y*h).shape)
        return (-y * np.log(h) - (1 - y)*np.log(1-h)).mean()

    def __sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))

    def activation(self, y):
        # Return the sigmoid function
        return self.__sigmoid(y)
    
    def predict(self, x):
        h = self.predict_proba(x)

        return h >= 0.5

    def predict_proba(self, x):
        #x = self.add_intercept(x)
        z = np.dot(x, self.w)
        h = self.activation(z)

        return h
    
    def gradient(self, x, y, h):
        N = len(y)

        if len(y.shape) == 1:
            y = np.expand_dims(y, 1)

        dw = np.dot(x.T, h - y) * (1.0 / N)
        db = (1.0/N)*(h - y)
        return dw, db