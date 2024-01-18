from LinearRegression import LinearRegressionModel, ModelArgs
import numpy as np

# Generate synthetic data
# x1 = np.linspace(0,10,100)
# x2 = np.linspace(-5,5,100)
# x = np.array([x1,x2]).reshape((100,2))
# y = 12.5*x1 + 25*x2 + 17.9
# y = y.reshape((100,1))

x = np.linspace(0,5,100).reshape((100,1))
y = (12.5*x + 21).reshape((x.shape[0],1))

args = ModelArgs(
    n_features = x.shape[1],
    lr = 0.01
)

model = LinearRegressionModel(args)
model.train(x, y, 1000)

print(model.w, model.b)