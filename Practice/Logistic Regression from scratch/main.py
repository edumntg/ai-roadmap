from LogisticRegression import LogisticRegressionModel, ModelArgs
import numpy as np
from sklearn.datasets import make_classification

# Generate synthetic data
X, y = make_classification(
    n_samples = 100,
    n_features = 2,
    n_redundant = 0,
    n_informative = 2,
    random_state = 1,
    n_clusters_per_class = 1
)

args = ModelArgs(
    n_features = X.shape[1],
    n_classes = 1,
    lr = 0.001
)

model = LogisticRegressionModel(args)
model.train(X, y, 1000)