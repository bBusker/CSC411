import part2
import part3
from part4 import alt_gen_set
from scipy.io import loadmat
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
from constants import *
import cPickle as pickle

def part6a():
    W = pickle.load(open("part5W.p", 'rb')) #load weights
    W_copy = W.copy()
    print W[w1, w1k]
    print W[w2, w2k]

    #picking 2 weights near the center, we will take the 14th pixel in the 11th row and the 14th pixel in the 17th
    #10 * 28 + 13 = 293
    #16 * 28 + 13 = 461 as our values of n, and any output may be chosen [0,9]

    dimension = 5

    M = loadmat("mnist_all.mat")
    x, y = alt_gen_set(M, 1)
    b = np.ones(shape = (K_NUM,1))

    X1 = np.linspace(-1, 1, dimension)
    X2 = np.linspace(-1, 1, dimension)

    res_a = np.zeros(shape = (dimension,dimension))
    max_deviation = 6
    for i in range(dimension):
        for j in range(dimension):
            W[w1, w1k] = W_copy[w1,w1k] + X1[i] * max_deviation 
            W[w2, w2k] = W_copy[w2,w2k] + X2[j] * max_deviation
            res_a[i, j] = part3.f(x,W,b,y)
            print "done " + str(i) + " " + str(j) + " cost: " + str(res_a[i,j]) + " with w values: " + str(W[w1, w1k]) + " " + str(W[w2,w2k])
            
    return W_copy[w1,w1k] + max_deviation*X1, W_copy[w2,w2k] + max_deviation*X2, res_a
    
K = 10
w1 = 362
w1k = 5
w2 = 360
w2k = 5

def part6bcd():
    M = loadmat("mnist_all.mat")
    W = pickle.load(open("part5W.p", 'rb')) #load weights
    x,y = alt_gen_set(M, 1)

    W[w1, w1k], W[w2, w2k] = -2,3
    W, res_b = grad_descent_2element(part3.f, part3.df, x, y, W, 0.1,20, 0, True)

    W[w1, w1k], W[w2, w2k] = -2,3
    W, res_c = grad_descent_2element(part3.f, part3.df, x, y, W, 0.1,20, 0.90, True)

    return res_b, res_c



def grad_descent_2element(f, df, x, y, init_W, alpha, _max_iter, momentum=0, printing=True):
    print("------------------------- Starting Grad Descent -------------------------")
    EPS = 1e-5  # EPS = 10**(-5)
    prev_t = init_W - 10 * EPS
    prev_grad = 0
    W = init_W.copy()
    V = np.zeros(shape = W.shape)
    max_iter = _max_iter
    iter = 0

    res = []
    while iter < max_iter: #and norm(t - prev_t) > EPS:
        prev_t = W.copy()
        res.append((W[w1, w1k], W[w2, w2k]))
        b = np.zeros(shape=(K_NUM,1))
        grad = df(x, y, W, b)
        V = momentum * V + alpha * grad
        # W -= alpha * grad
        W[w1, w1k] -= V[w1, w1k]
        W[w2, w2k] -= V[w2, w2k]
        if iter % 1 == 0 and printing:
            print("Iter %i: cost = %.5f" % (iter,  f(x, W, b, y)))
        elif iter % 50000 == 0:
            print("Training...")
        iter += 1

    print("Done!")
    return W, res

def part6():
    X1, X2, res = part6a()
    a_weights, b_weights = part6bcd()
    print a_weights
    print b_weights
    
    plt.contour(X1, X2, res)
    plt.autoscale(False) # To avoid that the scatter changes limits
    plt.plot([a for a, b in a_weights], [b for a,b in a_weights], 'yo-', label="No Momentum")
    plt.plot([a for a, b in b_weights], [b for a,b in b_weights], 'go-', label="Momentum")
    plt.legend(loc='upper left')
    fig = plt.gcf()
    fig.savefig('part5fig.png')
    plt.show()
    