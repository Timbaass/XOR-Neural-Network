import numpy as np
from NeuralNetworks.OneLayerNN import OneLayerNN
from NeuralNetworks.TwoLayerNN import TwoLayerNN

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

y = np.array([0, 1, 1, 0]).reshape(-1, 1)

OneLayerModel = OneLayerNN()
OneLayerModel.fit(X, y.flatten())

TwoLayerModel = TwoLayerNN()
TwoLayerModel.fit(X, y)

print("One Layer Model Predictions:", OneLayerModel.predict(X))

print("Two Layer Model Predictions:", TwoLayerModel.predict(X))

