from scipy import *
from scipy.io import loadmat
import part2
import os
import numpy as np

from constants import *


#PART 3
'''
x is (n * m)
'''
def f(x, W, b, y):
    p = part2.forward(x,W,b)
    return -1 * sum( dot(y.T, np.log(p)))

#PART 3a)
'''We look to write our gradient function with respect to each weight used in our neural network W_ij. In the slides we've 
provided an expression for the cost of the cost function with respect to the output pre-softmax. To obtain the change in 
cost w.r.t our weightings, we employ calculus getting dC/dW_ij = dC/do_i * do_i / dW_ij.

do_i / dW-ij is x_i
 '''

#PART 3b)
'''
We envision y and p to be (k * m) matrices and x to be (n * m) which would yield a (k * n) matrix, which we must transpose
to get our expected weightings gradient matrix of dimensions (n * k)
'''
def df(x,y,p):
    return np.matmul((y-p), x.T).T

#placeholder
def part3():

    #PART 3b) Verification Code
    M = loadmat("mnist_all.mat")

    #For our images, n = 7084, and our result ranges from 0 through 9, therefore k = 10.
    # W is (n * k)
    W = np.zeros(shape = (N_NUM,K_NUM))
    b = np.zeros(shape = (K_NUM,M_TRAIN))

    x = np.zeros(shape = (N_NUM, M_TRAIN))
    y = np.zeros(shape = (K_NUM, M_TRAIN))

    count = 0
    #Load our example
    for i in range(10):
        currSet = M["train" + str(i)].T / 255.0
        x[:, count: count + currSet.shape[1]] = currSet      
        y[i, count: count + currSet.shape[1]] = 1
        count += currSet.shape[1]

    cost = f(x,W,b,y)

    return 0