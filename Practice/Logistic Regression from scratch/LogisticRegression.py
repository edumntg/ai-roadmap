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

    def train(self, x, y, epochs = 10):

        x = self.add_intercept(x)

        for epoch in range(1, epochs + 1):
            # Calculate z (w1*x1+w2*x2+...+wn*xn + b)
            z = np.dot(x, self.w)

            # Compute probabilities
            h = self.__sigmoid(z)

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

            # Print
            print(f"Epoch {epoch}, acc: {acc:.2f}, loss: {l:.4f}")
    
    def add_intercept(self, x):
        intercept = np.ones((x.shape[0], 1))
        return np.concatenate((intercept, x), axis = 1)

    def loss(self, y, h):
        N = y.size
        return (-1.0/N)*np.sum((-y * np.log(h) - (1 - y)*np.log(1-h)))

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
        z = np.dot(x, self.w) + self.b
        h = self.activation(z)

        return h
    
    def gradient(self, x, y, h):
        N = len(y)
        dw = (1.0/N)*np.sum(np.dot(x.T, h - y))
        db = (1.0/N)*np.sum(h - y)
        return dw, db