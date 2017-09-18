import numpy as np

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    x_data = data[:,:-1]
    y_data = data[:,-1]

    new_y_data = np.zeros((y_data.shape[0], 10))

    for i,y in enumerate(y_data):
        new_y_data[i,int(y)] = 1

    return x_data, new_y_data