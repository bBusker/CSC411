from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import os
from scipy.io import loadmat

#Load the MNIST digit data
M = loadmat("mnist_all.mat")

#PART 3

#save 10 images, 1 from each digit
# for i in range(10):
#     imsave("part1/image" + str(i) + ".png", M["train"+str(i)][10].reshape((28,28)))

#PART 2

def compute(x, W, b)
    return softmax(tanh_layer(x, W, b))

def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))
    
def tanh_layer(y, W, b):    
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)

#PART 3
def f(x, W, b, y):
    p = compute(x,W,b)
    return -1 * (dot(y.T, np.log(p)))

#PART 3a)
'''We look to write our gradient function with respect to each weight used in our neural network W_ij. In the slides we've 
provided an expression for the cost of the cost function with respect to the output pre-softmax. To obtain the change in 
cost w.r.t our weightings, we employ calculus getting dC/dW_ij = dC/do_i * do_i / dW_ij.

do_i / dW-ij is x_i
 '''

#PART 3b)
'''
We envision y and p to be (k * m) matrices and x to be (m * n) which would yield a (k * n) matrix, which we must transpose
to get our expected weightings matrix of dimensions (n * k)
'''
def df(x,y,p):
    return np.matmul((y-p).T, x).T

#verification code
W = df



