import numpy as np
import cupy as cp 

# the derivative of Activation function shape is transpose of input shape

class Activation:
    def __init__(self):
        self.a = None
        self.deri = None

class Linear(Activation):
    def __call__(self, X):
        super().__init__()
        self.a = X
        self.deri = cp.ones_like(X).T

        return self.a

class Sigmoid(Activation):
    def __call__(self, X):
        super().__init__()
        self.a = 1.0/(1.0 + cp.exp(-X))
        self.deri = cp.multiply(self.a, 1-self.a).T

        return self.a
    
    def getName(self):
        return 'Sigmoid'
    
class Tanh(Activation):
    def __call__(self, X):
        super().__init__()
        self.a = cp.tanh(X)
        self.deri = (1 - self.a**2).T

        return self.a
    
    def getName(self):
        return 'Tanh'
    
class Softmax(Activation):
    def __call__(self, X):
        super().__init__()
        self.a = cp.exp(X - cp.max(X, axis=1, keepdims=True))/cp.sum(cp.exp(X - cp.max(X, axis=1, keepdims=True)), axis=1, keepdims=True)

        deriv = cp.zeros((X.shape[0], X.shape[1], X.shape[1]))
        for i in range(X.shape[0]):
            deriv[i] = cp.diag(self.a[i]) - cp.outer(self.a[i], self.a[i])
        self.deri = deriv.T

        return self.a
    
    def getName(self):
        return 'Softmax'
    
class Relu(Activation):
    def __call__(self, X):
        super().__init__()
        self.a = cp.maximum(0, X)
        self.deri = ((X >= 0).astype(int)).T

        return self.a
    
    def getName(self):
        return 'Relu'
    
    
if __name__ == '__main__':
    activation = Softmax()
    x = cp.array([[4.0,1.0,-1, 0], [1.0, 2.0, 3.0, 4.0]])
    print(activation(x))
    print(activation.deri)