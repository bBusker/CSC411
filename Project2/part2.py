from pylab import *
import numpy as np

# PART 2

def forward(x, W, b):
    t1 = dot(W.T, x)
    try:
        t2 = np.matmul(b, np.ones(shape = (1, x.shape[1])))
        t3 = t1 + t2
        t4 = softmax(t3)
    except:
        t2 = np.matmul(b, np.ones(shape = (1, 1)))
        t3 = t1 + t2.reshape((1,10))
        t4 = softmax(t3)
        t4 = softmax(t3.reshape((10,1)))
    return t4 #TODO: array dimensions of t4 when t2=1

    #return softmax(dot(W.T, x) + np.matmul(b, np.ones(shape=(1,x.shape[1]))))


def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an (k * m) matrix where k is the number of outputs for a single case, and m
    is the number of cases'''
    # e_y = np.exp(y - np.max(y))
    # return e_y / e_y.sum()
    return exp(y) / tile(sum(exp(y), 0), (len(y), 1))


def tanh_layer(x, W, b):
    '''Return the output of a tanh layer for the input matrix y. y
    is an (n * m) matrix where k is the number of inputs for a single case, and m
    is the number of cases and W is a (n * k) matrix'''
    return tanh(dot(W.T, x) + b)