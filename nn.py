import numpy as np
import sys

from load_data import load_data

"""
Implement a basic 784->100->10 neural network
"""

def ReLU(z):
    return np.maximum(z, 0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def softmax(z):
    S = np.sum(np.exp(z))

    return np.exp(z) / S

def softmax_prime(z):
    return softmax(z)

# xentropy for single activation - label pair
def cross_entropy_cost(a, y):
    return np.sum(np.nan_to_num(-y * np.log(a) - (1-y) * np.log(1-a)))

if __name__ == '__main__':

    filename = sys.argv[1]
    n_units = int(sys.argv[2])
    n_outputs = int(sys.argv[3])

    x_data, y_data = load_data(filename)

    # initializing

    n = x_data.shape[0]

    W = []
    B = []

    B.append(np.zeros(n_units))
    B.append(np.zeros(n_outputs))

    b = np.sqrt(6)/np.sqrt(x_data.shape[1] + n_units)
    W.append(np.random.rand(x_data.shape[1], n_units) * 2 * b - b)

    b = np.sqrt(6)/np.sqrt(n_units + n_outputs)
    W.append(np.random.rand(n_units, n_outputs) * 2 * b - b)

    totC = 0

    for x, y in zip(x_data, y_data):

        delta_B = []
        delta_W = []
        
        for b,w in zip(B,W):
            delta_B.append(np.zeros(b.shape))
            delta_W.append(np.zeros(w.shape))

        a = x
        activations = [x]
        zs = []

        i = 0

        for b,w in zip(B,W):
            i += 1

            z = np.matmul(w.transpose(),a) + b
            zs.append(z)

            if i < 2:
                a = sigmoid(z)
            else:
                a = softmax(z)

            activations.append(a)

        C = cross_entropy_cost(activations[-1], y)

        totC += C

        # output layer back prop
        delta = activations[-1] - y

        delta_W[-1] = np.multiply(
            np.tile(activations[-2], (len(delta), 1)).transpose(),
            delta
        )

        W[-1] = W[-1] - delta_W[-1]

    totC /= y_data.shape[0]
    print(np.max(W[-2]))