import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from Model import Sequential
from Layer import Dense, RNN, LSTM
from Activation import Sigmoid, Softmax, Tanh
from Loss import BinaryCrossEntropy, CrossEntropy
from Optimizer import GradientDescent


def CreateDataset(Samples, Classes, Feature, Informative, Redundant, Repeated, Seperate, Shift):
    X,Y = make_classification(
        n_samples= Samples,
        n_classes= Classes,
        n_features= Feature,
        n_informative= Informative,
        n_redundant= Redundant,
        n_repeated= Repeated,
        class_sep= Seperate,
        shift= Shift,
        n_clusters_per_class= 1
    )
    
    return X, np.expand_dims(Y, axis=1)

def ScatterPlot(X,Y):
    # color_map = ['r' if x==0 else 'b' for x in Y]
    plt.scatter(X[:,0], X[:,1], c=Y, alpha=0.8)


def DrawDecisionBoundary(W, b, ranged):
    for i in range(len(b)):
        a = -W[0][i]/W[1][i]
        y = [a*x - b[i]/W[1][i] for x in ranged]

        plt.plot(ranged, y, color='black')

def OneHotEncoding(Y):
    num_class = len(np.unique(Y))
    num_sample = len(Y)

    Y_one_hot = np.zeros((num_sample, num_class))

    for i, c in enumerate(Y.astype(int)):
        Y_one_hot[i,c] = 1

    return Y_one_hot

if __name__ == '__main__':
    # data_path = 'dataset.npy'
    # if len(data_path)==0:
    #     X,Y = CreateDataset(90, 3, 2, 2, 0, 0, 2, 1)
    #     np.save('dataset.npy', np.concatenate((X,Y), axis=1))

    # else:
    #     dataset = np.load(data_path)

    # X = dataset[:, 0:2]
    # Y = dataset[:, -1]
    # Y_one_hot = OneHotEncoding(Y)

    # X = np.array([
    #     [[1, 2], [3, 4], [5, 6]],  # First sequence
    #     [[7, 8], [9, 10], [11, 12]],  # Second sequence
    #     [[13, 14], [15, 16], [17, 18]],  # Third sequence
    #     [[19, 20], [21, 22], [23, 24]]   # Fourth sequence
    # ])
    # labels = np.array([1, 0, 2, 1])  
    # Y = np.eye(3)[labels]

    X = np.array([
        # [[0,1,0,0,0], [0,0,1,0,0], [1,0,0,0,0]],
        [[0,1,0,0,0], [0,1,0,0,0], [0,1,0,0,0]],
        [[0,0,1,0,0], [0,0,1,0,0], [1,0,0,0,0]],
    ])

    Y = np.array([
        # [0,0,0,1,0],
        [0,0,0,0,1],
        [0,0,0,0,1]
    ])

    model = Sequential([
        LSTM(output_shape=3, input_shape=(3,5), activation=Tanh(), return_sequences=False, truncated_step=3),
        Dense(output_shape=5, input_shape=3, activation=Softmax())
    ])


    # model.summary()
    model.compile(loss=CrossEntropy(), optimizer=GradientDescent())
    model.fit(X, Y, epochs=100, lr=0.5)
    print(model(X))
    # print(Y)


    # plt.ylim([np.min(X[:,0]), np.max(X[:,0])])
    # ScatterPlot(X,Y)
    # DrawDecisionBoundary(model.Layers[0].W, model.Layers[0].b, [-3,6])
    # plt.show()
