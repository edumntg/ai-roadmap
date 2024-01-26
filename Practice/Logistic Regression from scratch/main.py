from LogisticRegression import LogisticRegressionModel, ModelArgs
import numpy as np
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_classification(
    n_samples = 1000,
    n_features = 2,
    n_redundant = 0,
    n_informative = 2,
    random_state = 1,
    n_clusters_per_class = 1
)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

args = ModelArgs(
    n_features = X_train.shape[1],
    n_classes = 1,
    lr = 0.005
)

model = LogisticRegressionModel(args)
model.train(X_train, y_train, epochs = 1000, validation_dataset = (X_test, y_test))