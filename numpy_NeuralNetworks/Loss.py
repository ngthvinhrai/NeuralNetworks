import numpy as np

# the derivative of Loss function shape is transpose of input shape

class Loss:
    def __init__(self):
        self.loss = None
        self.deri = None

class BinaryCrossEntropy:
    def __call__(self, Y, Y_hat):
        N = Y.shape[0]
        self.loss = (-np.dot(Y.T, np.log(Y_hat)) - np.dot(1-Y.T, np.log(1-Y_hat)))/N
        self.deri = (-(Y.T-Y_hat.T)/(Y_hat.T * (1-Y_hat.T)))/N

        return self.loss[0]
    
class CrossEntropy(Loss):
    def __call__(self, Y, Y_hat):
        N = Y.shape[0]
        if Y_hat.ndim==2: self.loss = np.sum(-Y*np.log(Y_hat + 1e-3), axis=1).sum()/N
        else: self.loss = np.array([
            np.sum(-Y[i]*np.log(Y_hat[i] + 1e-3), axis=1).sum()/N for i in range(N)
        ]).sum()
        self.deri = (-Y.T/(Y_hat.T + 1e-3))/N

        return self.loss

class MeanSquareError(Loss):
    def __call__(self, Y, Y_hat):
        N = Y.shape[0]
        self.loss = ((Y - Y_hat)**2).sum()/N
        self.deri = -2*(Y.T - Y_hat.T)/N

        return self.loss

if __name__ == '__main__':
    Y = np.random.randn(10)
    Y_hat = np.random.randn(10)

    loss = MeanSquareError()
    loss(Y, Y_hat)
    print(loss.loss)
    print(loss.deri)

    