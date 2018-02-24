import part2
import part3
from part4 import alt_gen_set
from scipy.io import loadmat
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
from constants import *
import cPickle as pickle

def batch(x, y, batch_size, seed = 1):
    n = x.shape[1]

    np.random.seed(seed)
    indx = np.random.permutation(x.shape[1])
    
    x_batched = x[:, indx]
    y_batched = y[:, indx]

    for m in xrange(n):
        yield x_batched[:, m*batch_size:(m+1)*batch_size], y_batched[:,m*batch_size:(m+1)*batch_size]

def part6a():
    W, b = pickle.load(open("part5W.p", 'rb')) #load weights
    dimension = 15

    M = loadmat("mnist_all.mat")
    x, y = alt_gen_set(M, 1)

    orig1 = W[w1, w1k]
    orig2 = W[w2, w2k]

    print orig1 
    print orig2

    X1 = np.linspace(-5, 5, 10)
    X2 = np.linspace(-5, 5, 10)

    res_a = np.zeros(shape = (X1.size,X1.size))
    max_deviation = 1
    for i, w1v in enumerate(X1):
        for j, w2v in enumerate(X2):
            W[w1, w1k] = X1[i]
            W[w2, w2k] = X2[j]
            res_a[i, j] = part3.f(x,W,b,y)
            if True:
                print "done " + str(i) + " " + str(j) + " cost: " + str(res_a[i,j]) + " with w values: " + str(W[w1, w1k]) + " " + str(W[w2,w2k])
            
    return X1, X2, res_a
    
K = 10
w1 = 361
w1k = 5
w2 = 331
w2k = 5

def part6bcd():
    M = loadmat("mnist_all.mat")
    W, b = pickle.load(open("part5W.p", 'rb')) #load weights
    x,y = alt_gen_set(M, 1)

    W[w1, w1k], W[w2, w2k] = -3,-3
    W, res_b = grad_descent_2element(part3.f, part3.df, x, y,b, W, 0.3,20, 0, True)

    W[w1, w1k], W[w2, w2k] = -3,-3
    W, res_c = grad_descent_2element(part3.f, part3.df, x, y,b, W, 0.3,20, 0.7, True)

    return res_b, res_c



def grad_descent_2element(f, df, x, y, b, init_W, alpha, _max_iter, momentum=0, printing=True):
    print("------------------------- Starting Grad Descent -------------------------")
    gen = batch(x, y, 5000, 1)
    
    EPS = 1e-1  # EPS = 10**(-5)
    prev_t = init_W - 10 * EPS
    prev_grad = 0
    W = init_W.copy()
    V = np.zeros(shape = W.shape)
    max_iter = _max_iter
    iter = 0

    res = []
    while iter < max_iter: #and norm(t - prev_t) > EPS:
        xgen,ygen= next(gen)
        prev_t = W.copy()
        res.append((W[w1, w1k], W[w2, w2k]))
        grad, grad_W = df(xgen, ygen, W, b)
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
    X,Y, res = part6a()

    Y,X = np.meshgrid(X,Y)

    print argmin(res)

    a_weights, b_weights = part6bcd()
    # print a_weights
    # print b_weights
    
    CS = plt.contour(X, Y, res, label='Cost Function')
    plt.autoscale(False) # To avoid that the scatter changes limits
    plt.plot([a for a, b in a_weights], [b for a,b in a_weights], 'yo-', label="No Momentum")
    plt.plot([a for a, b in b_weights], [b for a,b in b_weights], 'go-', label="Momentum")
    plt.legend(loc='upper left')
    fig = plt.gcf()
    fig.savefig('part6badfig.png')
    plt.show()
    