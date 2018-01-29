from pylab import *
from scipy.misc import *


# Quadratic loss function
def quad_loss(x, y, theta, norm_const):
    # x = vstack((ones((1, x.shape[1])), x))
    return sum((y.T - dot(theta.T, x)) ** 2) / norm_const


# Gradient of quadratic loss function
def quad_loss_grad(x, y, theta, norm_const):
    # x = vstack((ones((1, x.shape[1])), x))
    return -2 * dot(x, (y.T - dot(theta.T, x)).T) / norm_const


# Generates corresponding x's, y's and thetas for gradient descent function
# Takes sorted input of training images and their corresponding training label
def generate_xyt(input_sets, labels):
    x = np.zeros((len(input_sets), 1025))
    try:
        y = np.zeros((len(input_sets), len(labels[0])))
        thetas = zeros((1025, len(labels[0])))
    except:
        y = np.zeros((len(input_sets), 1))
        thetas = zeros((1025, 1))
    for i in range(len(input_sets)):
        imdata = (imread("./cropped/" + input_sets[i][1]) / 255).reshape(1024)
        imdata = np.concatenate(([1], imdata))
        x[i] = imdata
        y[i] = labels[i]
    return x.T, y, thetas


# Gradient descent function. Taken from CSC411 website.
def grad_descent(f, df, x, y, init_t, alpha, _max_iter):
    print("------------------------- Starting Grad Descent -------------------------")
    EPS = 1e-5  # EPS = 10**(-5)
    prev_t = init_t - 10 * EPS
    t = init_t.copy()
    max_iter = _max_iter
    iter = 0
    while norm(t - prev_t) > EPS and iter < max_iter:
        prev_t = t.copy()
        grad = df(x, y, t, x.shape[1])
        t -= alpha * grad
        if iter % 1000 == 0:
            print("Iter %i: cost = %.2f" % (iter, f(x, y, t, x.shape[1])))
            # print("Gradient: ", grad, "\n")
        iter += 1
    return t


# Test a set of thetas for their accuracy on the test set
# Takes in a dictionary of test sets per actor, a dictionary of the corresponding correct answer for each actor,
# and the computed thetas
def test(test_sets, answers, thetas):
    correct = 0
    count = 0
    conc_sets = []
    print("------------------------- Testing -------------------------")
    for actor in test_sets:
        i = 0
        conc_sets += test_sets[actor]
        for image in test_sets[actor]:
            imdata = (imread("./cropped/" + image[1]) / 255).reshape(1024)
            imdata = np.concatenate(([1], imdata))
            prediction = np.dot(imdata, thetas)
            print("%s %i|pred:" % (actor, i), end=" ")
            print(prediction, end=" ")
            print("ans:", end=" ")
            print(answers[actor], end=" ")
            min = sum(abs(prediction - answers[actor]))
            guess = actor
            for answer in answers:
                if sum(abs(prediction - answers[answer])) < min:
                    min = sum(abs(prediction - answers[answer]))
                    guess = answer
            if guess == image[0]:
                correct += 1
                print("correct!")
            else:
                print("incorrect!")
            i += 1
            count += 1
    conc_sets_labels = [0 for i in range(len(conc_sets))]
    for i in range(len(conc_sets)):
        conc_sets_labels[i] = answers[conc_sets[i][0]]
    x, y, t = generate_xyt(conc_sets, conc_sets_labels)
    print("Cost: %.2f" % quad_loss(x, y, thetas, x.shape[1]))
    print("Score: %.2f" % (correct / count))
    return (correct / count)
