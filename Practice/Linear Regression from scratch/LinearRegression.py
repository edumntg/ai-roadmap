from dataclasses import dataclass
from simple_parsing.helpers import Serializable
import numpy as np

@dataclass
class ModelArgs(Serializable):
    n_features: int
    lr: float = 0.001

class LinearRegressionModel(object):
    
    def __init__(self, args: ModelArgs):
        self.args = args
        self.n_features = args.n_features
        self.lr = args.lr

        # Initialize weights and bias
        self.w = np.random.rand(self.n_features,1)
        self.b = 0

    def train(self, x, y, epochs = 10):

        for epoch in range(1, epochs + 1):
            # Calculate current estimation
            y_hat = np.dot(x, self.w) + self.b # size (n_samples, 1)

            # Calculate current loss
            loss = self.criterion(y, y_hat)

            # Calculate gradient
            dw, db = self.gradient(x, y, y_hat)

            # Update weights
            self.w -= self.lr*dw
            self.b -= self.lr*db

            # Print
            print(f"Epoch {epoch}, loss {loss:.4f}")
        pass

    def criterion(self, y, y_hat):
        N = len(y)
        return (1.0/N)*np.sum((y-y_hat)**2.0)
    
    def gradient(self, x, y, y_hat):
        N = len(y)
        dw = (-2.0/N)*np.sum(np.dot(x.T, y - y_hat))
        db = (-2.0/N)*np.sum(y - y_hat)
        return dw, db