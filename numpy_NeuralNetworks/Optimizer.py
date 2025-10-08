import numpy as np

class Optimizer:
    def __init__(self, norm=False):
        self.norm = norm

class GradientDescent(Optimizer):
    def __init__(self, norm=False):
        super().__init__(norm)

    def __call__(self, *weights):
        if self.norm: return [weight.T/np.linalg.norm(weight.T + 1e-5, axis=0) for weight in weights]
        else: return [weight.T for weight in weights]
    
class Momentum(Optimizer):
    def __init__(self, beta=0.9, norm=False):
        super().__init__(norm)
        self.beta = beta
        self.v = None

    def __call__(self, *weights):
        if self.v is None:
            self.v = [np.zeros(weight.T.shape) for weight in weights]
        self.v = [self.beta * v + (1 - self.beta) * weight.T for v, weight in zip(self.v, weights)]
        
        if self.norm: return [v/np.linalg.norm(v + 1e-5, axis=0) for v in self.v]
        else: return self.v
    
if __name__ == '__main__':
    X = np.array([[1,2,3],[4,5,6]])
    b = np.array([2])
    X, b = GradientDescent(norm=True)(X,b)
    print(X, b) 