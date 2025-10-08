from NeuralNetworks.numpy_NeuralNetworks.Model import Sequential
from NeuralNetworks.numpy_NeuralNetworks.Layer import Dense
from NeuralNetworks.numpy_NeuralNetworks.Activation import Linear, Relu, Softmax

def main():
    model = Sequential([
        Dense(output_shape=1, input_shape=1, activation=Linear())
    ])

    

if __name__ == "__main__":
    main()