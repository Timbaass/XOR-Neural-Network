import numpy as np

class TwoLayerNN:
    def __init__(self, lr=0.1 , epochs=10000, seed=2):
        np.random.seed(seed)
        self.lr = lr
        self.epochs = epochs

        # 2 -> 2 -> 1 mimarisi
        self.W1 = np.random.randn(2, 2) * np.sqrt(1/2)
        self.b1 = np.zeros((1, 2))
        self.W2 = np.random.randn(2, 1) * np.sqrt(1/2)
        self.b2 = np.zeros((1, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m = X.shape[0]

        for e in range(self.epochs + 1):

            # ----- FORWARD -----
            Z1 = np.dot(X, self.W1) + self.b1      # (m,2)
            A1 = self.sigmoid(Z1)
            Z2 = np.dot(A1, self.W2) + self.b2     # (m,1)
            A2 = self.sigmoid(Z2)

            # ----- LOSS -----
            loss = -np.mean(y * np.log(A2 + 1e-9) + (1 - y) * np.log(1 - A2 + 1e-9))

            # ----- BACKWARD -----
            dZ2 = A2 - y
            dW2 = np.dot(A1.T, dZ2) / m
            db2 = np.sum(dZ2, axis=0, keepdims=True) / m

            dZ1 = np.dot(dZ2, self.W2.T) * A1 * (1 - A1)
            dW1 = np.dot(X.T, dZ1) / m
            db1 = np.sum(dZ1, axis=0, keepdims=True) / m

            # ----- UPDATE -----
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

            if e % 1000 == 0:
                print(f'{e}.epoch Loss:', loss)

    def predict(self, X):
        A1 = self.sigmoid(np.dot(X, self.W1) + self.b1)
        A2 = self.sigmoid(np.dot(A1, self.W2) + self.b2)
        return (A2 > 0.5).astype(int).flatten()
