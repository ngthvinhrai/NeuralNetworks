import time
import numpy as np
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
    
    def backward(self, lr):
        dL_A = self.loss.deri
        for i in reversed(range(len(self.Layers))): 
            dL_A, *grad = self.Layers[i].backward(dL_A, self.optimizer[i], lr)

    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = [copy.deepcopy(optimizer) for _ in range(len(self.Layers))]

    def fit(self, X, Y, val_data=None, batch_size=1, epochs=1, lr=0.1):
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        lenght = 50
        step = lenght / len(X)
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}:\t' + '[' + '-'*lenght +']', end='')
            val_evaluate = ""
            loss = 0
            indices = list(range(len(X)))
            np.random.shuffle(indices)
            
            for i in range(0, len(X), batch_size):
                if i+batch_size > len(X): batch_idx = indices[i:len(X)]
                else: batch_idx = indices[i:i+batch_size]

                self.forward(X[batch_idx])
                Y_hat = self.Layers[-1].getOutput()
                loss += self.loss(Y[batch_idx], Y_hat)
                self.backward(lr)

                progress = int((i+len(Y_hat)) * step)
                print(f'\rEpoch {epoch+1}/{epochs}:\t' + '[' + '='*progress + '>' + '-'*(lenght - progress) +']', end='')
                print(f' {i+len(Y_hat)}/{len(X)}', end='')
            
            try:
                count = np.bincount(np.argmax(self.predict(X), axis=1) == np.argmax(Y, axis=1))
            except np.exceptions.AxisError:
                count = np.bincount(np.argmax(self.predict(X), axis=1) == Y)
            history['loss'].append(loss/(len(X)/batch_size))
            history['accuracy'].append(count[1]/len(X))

            if val_data != None:
                self.forward(val_data[0])
                val_loss = self.loss(val_data[1], self.Layers[-1].getOutput())
                try:
                    val_count = np.bincount(np.argmax(self.predict(val_data[0]), axis=1) == np.argmax(val_data[1], axis=1))
                except np.exceptions.AxisError:
                    val_count = np.bincount(np.argmax(self.predict(val_data[0]), axis=1) == val_data[1])
                val_evaluate = f' - val_loss: {val_loss:.4f} - val_accuracy: {val_count[1]/len(val_data[0]):.4f}'
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_count[1]/len(val_data[0])) 

            print(f' - loss: {history['loss'][-1]:.4f} - accuracy: {history['accuracy'][-1]:.4f}' + val_evaluate)

        return history
    
    def evaluate(X, Y):
        try:
            count = np.bincount(np.argmax(self.predict(X), axis=1) == np.argmax(Y, axis=1))
        except np.exceptions.AxisError:
            count = np.bincount(np.argmax(self.predict(X), axis=1) == Y)
        
        print("="*30 + "EVALUATION" + "="*30)
        print(f"accuracy: {count[1]/len(X)}")
            
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
    print(np.array([[1,3,4,4]]))