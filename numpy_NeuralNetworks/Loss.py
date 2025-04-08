import numpy as np

# the derivative of Loss function shape is transpose of input shape

class Loss:
    def __init__(self):
        self.loss = None
        self.deri = None

class BinaryCrossEntropy:
    def __call__(self, Y, Y_hat):
        super().__init__()
        N = Y.shape[0]
        self.loss = (-np.dot(Y.T, np.log(Y_hat)) - np.dot(1-Y.T, np.log(1-Y_hat)))/N
        self.deri = (-(Y.T-Y_hat.T)/(Y_hat.T * (1-Y_hat.T)))/N

        return self.loss[0]
    
class CrossEntropy(Loss):
    def __call__(self, Y, Y_hat):
        super().__init__()
        N = Y.shape[0]
        if Y_hat.ndim==2: self.loss = np.trace(-np.dot(Y.T, np.log(Y_hat + 1e-3))) / N
        else: self.loss = np.array([
            np.trace(-np.dot(Y[i].T, np.log(Y_hat[i] + 1e-3))) / N for i in range(N)
        ]).sum()
        self.deri = (-Y.T/(Y_hat.T + 1e-3))/N

        return self.loss
    
if __name__ == '__main__':
    # Y = np.array([[1,0,0], [0,0,1]])
    # Y_hat = np.array([[0.1, 0.9, 0.5], [0.4, 0.8, 0.7]])

    Y = np.random.randint(0, 2, (3, 2, 3))  # Binary class labels
    Y_hat = np.random.randn(3, 2, 3) + 5  # Predicted probabilities (should be in range 0-1)

    loss = CrossEntropy()
    loss(Y, Y_hat)
    print(loss.loss)

    