###### copied from mp1 ######

import numpy as np

class GD_Armijo(object):
    def __init__(self, Q, A, b, epsilon, alpha0, sigma, beta):
        # arguments for f(x) (L_c(x, lambda))
        self.Q = Q # n*n pd matrix
        self.A= A # m*n matrix
        self.b = b # m*1 vector

        # arguments for Armijo's rule
        self.epsilon = epsilon
        self.alpha0 = alpha0
        self.sigma = sigma
        self.beta = beta

        self.lambda_k = None # lambda is a m*1 vector
        self.c_k = None # c is a number

    # function for h(x) = Ax - b
    # x: n*1 vector
    # A: m*n matrix
    # b: m*1 vector
    # returns a m*1 vector
    def h(self, x):
        return np.matmul(self.A, x) - self.b

    def norm_h(self, x):
        # transform m*1 vector to 1d array with length m
        hx = self.h(x).T[0]
        return np.linalg.norm(hx)

    # function for f(x), returns a number
    # in this mp, f is L(x, c, lambda)
    def f(self, x):
        fx = np.matmul(np.matmul(x.T, self.Q), x)
        hx = self.h(x)
        lambda_hx = np.matmul(self.lambda_k.T, hx)
        result = fx[0, 0] + lambda_hx[0, 0] + self.c_k * self.norm_h(x)**2
        return result

    # function for dL(x, c, lambda)/dx, returns an n*1 vector
    def grad_f(self, x):
        dfx = np.matmul(self.Q.T + self.Q, x)
        dgx = np.matmul(self.lambda_k.T, self.A).T
        dgx_norm = np.matmul(self.A.T, self.h(x))
        return dfx + dgx + self.c_k * dgx_norm

    # Armijo's rule for finding alpha given x_k and other parameters
    def armijo(self, x, alpha0, sigma, beta):
        alpha = alpha0
        fx = self.f(x)
        dk = -self.grad_f(x)
        grad_f_dk = np.matmul(self.grad_f(x).T, dk)
        sigma_grad_f_dk = sigma * grad_f_dk[0, 0]
        stop = False
        while not stop:
            lhs = self.f(x + alpha * dk)
            rhs = fx + alpha * sigma_grad_f_dk
            if lhs <= rhs:
                stop = True
            else:
                alpha = alpha * beta
        return alpha

    def gradient_descent(self, x, lambda_k, c_k):
        self.lambda_k = lambda_k
        self.c_k = c_k
        i = 0
        while abs(np.linalg.norm(self.grad_f(x))) >= self.epsilon:
            alpha = self.armijo(x, self.alpha0, self.sigma, self.beta)
            x = x - alpha * self.grad_f(x)
            i = i + 1
            # print(x)
        return x, self.f(x), i