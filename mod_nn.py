import numpy as np
import sys

from load_data import load_data

"""
Implement a basic 784->100->10 neural network
"""

def ReLU(z):
    return np.maximum(z, 0)

def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    print(a)
    return a

def sigmoid_prime(z, y):
    return sigmoid(z)*(1-sigmoid(z))

def softmax(z):
    S = np.sum(np.exp(z), axis=1)

    a = np.divide(np.exp(z), np.tile(S,(z.shape[1], 1)).transpose())

    return a

def softmax_prime(z, y):
    return (y - np.multiply(y, softmax(z)))

def cross_entropy_cost(a, y):
    return np.sum(np.nan_to_num(np.multiply(-y, np.log(a)) - np.multiply((1-y), np.log(1-a))))

def forwardPropagation(x, y, params):

    W = params['weights']
    B = params['biases']
    a_funcs = params['activation_functions']

    n_layers = len(W)

    a = x
    activations = [x]
    zs = []

    for i in range(n_layers):
        z = np.matmul(a, W[i]) + B[i]
        zs.append(z)

        a = a_funcs[i][0](z)
        activations.append(a)

    return activations, zs

# back prop for cross entropy loss
def backPropagation(x,y,params):

    W = params['weights']
    B = params['biases']

    A = params['activations']
    Z = params['preActivations']

    a_funcs = params['activation_functions']
    eta = params['learning_rate']

    N = int(x.shape[0]) # batch size

    deltas = []
    delta_B = []
    delta_W = []

    for b,w in zip(B,W):
        deltas.append(np.zeros(b.shape))
        delta_B.append(np.zeros(b.shape))
        delta_W.append(np.zeros(w.shape))

    # output layer
    delta = a_funcs[-1][1](Z[-1], y)
    delta /= N

    deltas[-1] = delta
    delta_W[-1] = np.matmul(A[-2].transpose(), delta)
    delta_B[-1] = np.sum(delta, axis=0)

    for i in range(2, len(A)):
        delh = np.matmul(deltas[-(i-1)], W[-(i-1)].transpose())
        delta = a_funcs[-i][1](A[-i], y)
        delta = np.multiply(delh, delta)

        delta /= N

        deltas[-i] = delta
        delta_W[-i] = np.matmul(A[-(i+1)].transpose(), delta)
        delta_B[-i] = np.sum(delta, axis=0)

    for i in range(len(A) - 1):
        W[i] -= eta * delta_W[i]
        B[i] -= eta * delta_B[i]

    return W, B

def trainNeuralNetwork(X, Y, W, B, a_funcs, n_epochs=200, batch_size=32, eta=0.01):


    batch_order = np.random.permutation(X.shape[0])

    for i in range(n_epochs):

        for n in range(int(np.ceil(X.shape[0] / batch_size)) - 1):

            M = np.minimum((n + 1) * (batch_size), X.shape[0] - 1)
            batch_idx = batch_order[n * batch_size:M]

            Xb = X[batch_idx,:]
            Yb = Y[batch_idx]

            params = {}

            params['weights'] = W
            params['biases'] = B
            params['activation_functions'] = a_funcs
            params['learning_rate'] = eta

            A, Z = forwardPropagation(Xb, Yb, params)
            params['activations'] = A
            params['preActivations'] = Z

            W, B = backPropagation(Xb, Yb, params)
            params['weights'] = W
            params['biases'] = B

            AM, ZM = forwardPropagation(Xb, Yb, params)
            cost = cross_entropy_cost(AM[-1], Yb)

            #print('Epoch: {0}\tCost:{1}\t\t'.format((i+1), cost), end='\r')
            #print(AM[-1], Yb)

    return cost, W, B

if __name__ == '__main__':

    filename = sys.argv[1]
    n_units = int(sys.argv[2])
    n_outputs = int(sys.argv[3])

    x_data, y_data = load_data(filename)

    # initializing

    n = x_data.shape[0]

    n_epochs = 200

    W = []
    B = []

    B.append(np.zeros(n_units))
    B.append(np.zeros(n_outputs))

    b = np.sqrt(6)/np.sqrt(x_data.shape[1] + n_units)
    W.append(np.random.rand(x_data.shape[1], n_units) * 2 * b - b)

    b = np.sqrt(6)/np.sqrt(n_units + n_outputs)
    W.append(np.random.rand(n_units, n_outputs) * 2 * b - b)

    C, weights, biases = trainNeuralNetwork(x_data, y_data, W, B,
        a_funcs=[[sigmoid, sigmoid_prime], [softmax, softmax_prime]])