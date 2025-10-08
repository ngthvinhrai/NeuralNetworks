import time
import numpy as np
import cupy as cp
import copy
import os
import pickle
import json

class Sequential:
    def __init__(self, Layers=[]):
        self.Layers = Layers

    def summary(self):
        header = ['Layer', 'Output Shape', 'Num Params']
        print(f"{header[0]:<30} {header[1]:<20} {header[2]:<20}")
        print("-" * 60)

        for Layer in self.Layers:
            print(f'{Layer.getName():<30} {Layer.output_shape:<20} {(Layer.output_shape + 1)*Layer.input_shape:<20}')
        print("-" * 60)

    def add(self, Layer):
        if len(self.Layers) != 0: Layer.input_shape = self.Layers[-1].output_shape
        self.Layers.append(Layer)

    def predict(self, X):
        self.forward(X)
        return self.Layers[-1].getOutput()
    
    def forward(self, X):
        A = X
        for Layer in self.Layers:
            A = Layer.forward(A)
    
    def backward(self, lr, gradient_norm):
        dL_A = self.loss.deri
        for i in reversed(range(len(self.Layers))): 
            dL_A = self.Layers[i].backward(dL_A, self.optimizer[i], lr, gradient_norm)

    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = [copy.deepcopy(optimizer) for _ in range(len(self.Layers))]

    def evaluate(self, X, Y, val_data=None, batch_size=0, his=None):
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []} if his is None else his
        loss = 0
        val_evaluate = ""
        count = 0
        for i in range(0, len(X), batch_size):
            if i == (len(X)//batch_size)*batch_size and len(X)-i != 0:
                self.forward(X[i: len(X)])
                Y_hat = self.Layers[-1].getOutput()
                loss += self.loss(Y[i: len(X)], Y_hat)
                count += cp.sum((cp.argmax(Y_hat, axis=1) == cp.argmax(Y[i:len(X)], axis=1)).astype(int))
            else:
                self.forward(X[i: i+batch_size])
                Y_hat = self.Layers[-1].getOutput()
                loss += self.loss(Y[i: i+batch_size], Y_hat)
                count += cp.sum((cp.argmax(Y_hat, axis=1) == cp.argmax(Y[i: i+batch_size], axis=1)).astype(int))

        history['loss'].append(loss/(len(X)/batch_size))
        history['accuracy'].append(count/len(X))
        if val_data != None:
            self.forward(val_data[0])
            Y_hat = self.Layers[-1].getOutput()
            val_loss = self.loss(val_data[1], Y_hat)
            val_count = cp.bincount(cp.argmax(Y_hat, axis=1) == cp.argmax(val_data[1], axis=1))
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_count[1]/len(val_data[0]))
            val_evaluate = ' - val_loss: %.4f - val_accuracy: %.4f'%(history['val_loss'][-1], history['val_accuracy'][-1])

        print(' - loss: %.4f - accuracy: %.4f'%(history['loss'][-1], history['accuracy'][-1]) + val_evaluate) 

        return history
        

    def fit(self, X, Y, val_data=None, batch_size=0, epochs=1, lr=0.1, gradient_norm=False):
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        lenght = 50
        step = lenght / len(X)
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}:\t' + '[' + '-'*lenght +']', end='')

            for i in range(0, len(X), batch_size):
                if i == (len(X)//batch_size)*batch_size and len(X)-i != 0:
                    self.forward(X[i: len(X)])
                    Y_hat = self.Layers[-1].getOutput()
                    loss = self.loss(Y[i: len(X)], Y_hat)
                    batch_count = len(X)
                else:
                    self.forward(X[i: i+batch_size])
                    Y_hat = self.Layers[-1].getOutput()
                    loss = self.loss(Y[i: i+batch_size], Y_hat)
                    batch_count = i+batch_size
                          
                progress = int(batch_count * step)
                print(f'\rEpoch {epoch+1}/{epochs}:\t' + '[' + '='*progress + '>' + '-'*(lenght - progress) +']', end='')
                print(f'{batch_count}/{len(X)}', end='')
                print(f' - loss: {loss:.4f}', end='')

                self.backward(lr, gradient_norm)      

            print(f'\rEpoch {epoch+1}/{epochs}:\t' + '[' + '='*progress + '>' + '-'*(lenght - progress) +']', end='')

            history = self.evaluate(X, Y, val_data, 32, his=history)

        return history
            
    def save_weights(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        os.chdir(path)

        for i, Layer in enumerate(self.Layers):
            if not os.path.exists(Layer.__class__.__name__ + f'{i}'):
                os.mkdir(Layer.__class__.__name__ + f'{i}')
            os.chdir(Layer.__class__.__name__ + f'{i}')
            Layer.save()
            os.chdir('..')

    def save_model(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        os.chdir(path)

        return

        for i, Layer in enumerate(self.Layers):
            with open(Layer.__class__.__name__ + f'{i}' + '.json', "w") as f:
                json.dump(Layer, f)

    def load_weight(self, path):
        os.chdir(path)
        for i, Layer in enumerate(self.Layers):
            os.chdir(Layer.__class__.__name__ + f'{i}')
            Layer.load()
            os.chdir('..')

    def __call__(self, X):
        self.forward(X)
        return self.Layers[-1].getOutput()

                


if __name__ == '__main__':

    # model = Sequential([
    #     Dense(3, 4, Sigmoid()),
    #     Dense(2,3, Sigmoid()),
    #     Dense(1,2, Sigmoid())
    # ])
    # model.summary()
    p = "s"
    p = p.join(["af"])
    print(p)