import part2
import part3
from scipy.io import loadmat
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
from constants import *
import cPickle as pickle

# def grad_descent(f, df, x, y, init_W, alpha, _max_iter, momentum=0, printing=True):
#     print("------------------------- Starting Grad Descent -------------------------")
#     EPS = 1e-5  # EPS = 10**(-5)
#     prev_t = init_W - 10 * EPS
#     prev_grad = 0
#     W = init_W.copy()
#     V = np.zeros(shape = W.shape)
#     max_iter = _max_iter
#     iter = 0
#
#     while iter < max_iter: #and norm(t - prev_t) > EPS:
#         prev_t = W.copy()
#         b = np.zeros(shape=(K_NUM,1))
#         grad = df(x, y, W, b)
#         V = momentum * V + alpha * grad
#         # W -= alpha * grad
#         W -= V
#         if iter % 100 == 0 and printing:
#             print("Iter %i: cost = %.5f" % (iter,  f(x, W, b, y)))
#         elif iter % 50000 == 0:
#             print("Training...")
#         iter += 1
#         prev_grad = grad
#
#     print("Done!")
#     return W

def grad_descent(f, df, x, y, init_W, alpha_w, alpha_b, _max_iter, momentum = 0, printing=True):
    print("------------------------- Starting Grad Descent -------------------------")
    EPS = 1e-5  # EPS = 10**(-5)
    prev_t = init_W - 10 * EPS
    W = init_W.copy()
    b = np.zeros(shape=(K_NUM, 1))
    V_w = np.zeros(shape = W.shape)
    V_b = np.zeros(shape = b.shape)
    max_iter = _max_iter
    iter = 0

    while iter < max_iter: #and norm(t - prev_t) > EPS:
        #prev_t = W.copy()
        grad_w, grad_b = df(x, y, W, b)
        V_w = momentum * V_w + alpha_w * grad_w
        W -= V_w
        V_b = momentum * V_b + alpha_b * grad_b
        b -= V_b
        if iter % 100 == 0 and printing:
            print("Iter %i: cost = %.5f" % (iter,  f(x, W, b, y)))
        elif iter % 50000 == 0:
            print("Training...")
        iter += 1

    print("Done!")
    return W, b


def generate_sets(database, size):
    train_set = np.zeros((size, N_NUM))
    val_set = np.zeros((6000, N_NUM))
    train_set_sol = np.zeros((size, K_NUM))
    val_set_sol = np.zeros((6000, K_NUM))
    count = np.zeros((10,1))

    for i in range(6000):
        rand_dgt = np.random.random_integers(0,9)
        val_set[i] = database["train"+str(rand_dgt)][int(count[rand_dgt][0])] / 255.0
        val_set_sol[i][rand_dgt] = 1
        count[rand_dgt] += 1

    for i in range(size):
        rand_dgt = np.random.random_integers(0,9)
        train_set[i] = database["train"+str(rand_dgt)][int(count[rand_dgt][0])] / 255.0
        train_set_sol[i][rand_dgt] = 1
        count[rand_dgt] += 1

    return train_set.T, train_set_sol.T, val_set.T, val_set_sol.T


def alt_gen_set(database, scale):
    x = np.zeros(shape = (N_NUM, M_TRAIN))
    y = np.zeros(shape = (K_NUM, M_TRAIN))

    count = 0
    #Load our example
    for i in range(10):
        currSet = database["train" + str(i)].T / 255.0
        x[:, count: count + currSet.shape[1]] = currSet      
        y[i, count: count + currSet.shape[1]] = 1
        count += currSet.shape[1] 

    x = x[:, 0:M_TRAIN / scale]
    y = y[:, 0:M_TRAIN / scale]

    return x,y

def test(database, size, W, b):
    test_set = []
    correct = 0

    for i in range(size):
        rand_dgt = np.random.random_integers(0, 9)
        test_set += [database["test"+str(rand_dgt)][i]]
        guess = part2.forward(test_set[i], W, b)
        if guess == rand_dgt:
            correct += 1

    return correct/size

def test2(database, solutions, W, b):
    correct = 0
    database = database.T
    solutions = solutions.T

    for i in range(len(database)):
        guess = part2.forward(database[i], W, b)
        if np.argmax(guess) == np.argmax(solutions[i]):
            correct += 1

    return correct/float(len(database))


def part5(alpha_w, alpha_b, _max_iter, printing):
    M = loadmat("mnist_all.mat")

    results_val = []
    results_train = []
    x = []
    train_set, train_set_sol, val_set, val_set_sol = generate_sets(M, 40000)
    W_init = np.random.rand(784, 10)

    for i in range(100, 500, 100):
        print(i)
        # train_set, sol_set = alt_gen_set(M, 1)
        W, b = grad_descent(part3.f, part3.df, train_set, train_set_sol, W_init, alpha_w, alpha_b, i, printing)
        # b = np.ones((10, 1))*100
        # print(W, b)
        # print("Testing... {}% correct".format(test(M, 500, W, b)*100))
        results_val += [test2(val_set, val_set_sol, W, b)]
        results_train += [test2(train_set, train_set_sol, W, b)]
        x += [i]

    pickle.dump( W, open( "part5W_shichen.p", "wb" ) )
    plt.plot(x, results_val)
    plt.plot(x, results_train)
    plt.legend(["Validation Set Accuracy", "Training Set Accuracy"])
    plt.title("Part 5 Learning Curve")
    plt.show()