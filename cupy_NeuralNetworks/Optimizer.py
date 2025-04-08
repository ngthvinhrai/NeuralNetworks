import numpy as np
import cupy as cp 

class GradientDescent:
    def __call__(self, *weights):
        return [weight.T for weight in weights]
    
class Momentum:
    def __init__(self, beta=0.9):
        self.beta = beta
        self.v = None

    def __call__(self, *weights):
        if self.v is None:
            self.v = [cp.zeros(weight.T.shape) for weight in weights]
        self.v = [self.beta * v + (1 - self.beta) * weight.T for v, weight in zip(self.v, weights)]
        return self.v
    
if __name__ == '__main__':
    X = cp.array([[1,2,3],[4,5,6]])
    b = cp.array([2])
    X, b = GradientDescent()(X,b)
    print(X, b) 