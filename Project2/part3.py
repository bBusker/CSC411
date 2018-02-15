from scipy import *
import part2


#PART 3
def f(x, W, b, y):
    p = part2.part2(x,W,b)
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

#placeholder
def part3():
    return 0