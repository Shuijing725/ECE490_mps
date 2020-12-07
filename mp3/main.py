import numpy as np
from gd_armijo import GD_Armijo

# m = 2, n = 3
Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
A = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[1], [1]])
epsilon=0.00001
alpha0=0.001
sigma=0.1
beta=0.1

# initialize the gradient descent optimizer for L(x, c, lambda)
minimizer = GD_Armijo(Q, A, b, epsilon, alpha0, sigma, beta)

# given ck, calculate c(k+1)
# method: whether we are using method 1, 2, 3, 4 in Hint
# 1: const, 2: add, 3: multiply, 4: multiply_with_condition
def update_c(c, method='const', hx=None, last_hx=None):
    if method == 'const':
        newc = c
    elif method == 'add':
        newc = c + 0.5
    elif method == 'multiply':
        newc = c * 1.1
    else:
        if not hx or not last_hx:
            raise ValueError('For method 4, need h(xk) and h(x(k-1))!')
        if np.linalg.norm(hx) > 0.9 * np.linalg.norm(last_hx):
            newc = c * 1.1
        else:
            newc = c
    return newc

# main function for method of multipliers
def method_of_multiplier(epsilon, x, c_k, lambda_k):
    i = 0
    while minimizer.norm_h(x) >= epsilon:
        x, _, _ = minimizer.gradient_descent(x, lambda_k, c_k)
        lambda_k = lambda_k + c_k * minimizer.h(x)
        c_k = update_c(c_k)
        i = i + 1
    return x, i

x0 = np.array([[1], [0], [0]])
lambda0 = np.array([[1], [0]])
x, num_iter = method_of_multiplier(epsilon=0.00001, x=x0, c_k=1, lambda_k=lambda0)
print('solution x:', x, '\n, number of iterations:', num_iter)

# Todo: 1. add plotting for part3; 2. check with scipy optimize