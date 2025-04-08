import pandas as pd
import numpy as np
import random
from Model import Sequential
from Layer import RNN, LSTM, Dense
from Loss import CrossEntropy
from Activation import Tanh, Softmax, Sigmoid
from Optimizer import GradientDescent

def parity_check_dataset(num_samples=60):
    X = []
    Y = []
    for i in range(num_samples):
        X.append([])
        a = [random.randint(0, 1) for _ in range(5)]
        X[i].append(a)
        b = [random.randint(0, 1) for _ in range(5)]
        X[i].append(b)
        total = int("".join(map(str, a)), 2) + int("".join(map(str, b)), 2)
        Y.append([1, 0] if total % 2 == 0 else [0, 1])
    return np.array(X), np.array(Y)


if __name__ == '__main__':
    # data = pd.read_csv('data/rnn_multiclass_classification_dataset.csv')
    X, Y = parity_check_dataset()

    model = Sequential([
        LSTM(output_shape=64, input_shape=(2,5), activation=Tanh(), return_sequences=False, truncated_step=2),
    ])
    model.add(Dense(output_shape=2, activation=Softmax()))

    # model.summary()
    model.compile(loss=CrossEntropy(), optimizer=GradientDescent())
    model.fit(X[:50], Y[:50], batch_size=10, epochs=100, lr=0.3)
    print(model.predict(X[:50]) == Y[:50])


    # # Y = np.eye(3)[data['Label'].to_numpy()]
    # X = data.drop('Label', axis=1).to_numpy()
    # X = np.expand_dims(X, axis=2)