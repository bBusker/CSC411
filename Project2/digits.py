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

import mnist

#Load the MNIST digit data
M = loadmat("mnist_dataset.mat")

#PART 1

#save 10 images, 1 from each digit
# for i in range(10):
#     imsave("part1/image" + str(i) + ".png", M["train"+str(i)][10].reshape((28,28)))

#PART 2

def part2(x, W, b):
    return mnist.softmax(mnist.tanh_layer(x, W, b))

#PART 3
def f(x, W, b, y):
    p = part2(x, W, b)
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



