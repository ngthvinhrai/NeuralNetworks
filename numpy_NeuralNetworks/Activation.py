import numpy as np

# the derivative of Activation function shape is transpose of input shape

class Activation:
    def __init__(self):
        self.a = None
        self.deri = None

class Linear(Activation):
    def __call__(self, X):
        super().__init__()
        self.a = X
        self.deri = np.ones_like(X).T

        return self.a

class Sigmoid(Activation):
    def __call__(self, X):
        super().__init__()
        self.a = 1.0/(1.0 + np.exp(-X))
        self.deri = np.multiply(self.a, 1-self.a).T

        return self.a
    
    def getName(self):
        return 'Sigmoid'
    
class Tanh(Activation):
    def __call__(self, X):
        super().__init__()
        self.a = np.tanh(X)
        self.deri = (1 - self.a**2).T

        return self.a
    
    def getName(self):
        return 'Tanh'
    
class Softmax(Activation):
    def __call__(self, X):
        super().__init__()
        self.a = np.exp(X - np.max(X, axis=1, keepdims=True))/np.sum(np.exp(X - np.max(X, axis=1, keepdims=True)), axis=1, keepdims=True)

        deriv = np.zeros((X.shape[0], X.shape[1], X.shape[1]))
        for i in range(X.shape[0]):
            deriv[i] = np.diag(self.a[i]) - np.outer(self.a[i], self.a[i])
        self.deri = deriv.T

        return self.a
    
    def getName(self):
        return 'Softmax'
    
class Relu(Activation):
    def __call__(self, X):
        super().__init__()
        self.a = np.maximum(0, X)
        self.deri = ((X >= 0).astype(int)).T

        return self.a
    
    def getName(self):
        return 'Relu'
    
    
if __name__ == '__main__':
    activation = Softmax()
    x = np.array([[4.0,1.0,-1, 0], [1.0, 2.0, 3.0, 4.0]])
    print(activation(x))
    print(activation.deri)