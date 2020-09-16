import numpy as np
# Choose n, then generate random values for Q (positive definite),b, and c
# n=5
# Q=np.random.random((n,n))
# Q=Q.dot(Q.T)+0.1*np.identity(n)
# b=np.random.random(n)
# c=np.random.random(1)
# np.savetxt('Q.txt', Q)
# np.savetxt('b.txt', b)
# np.savetxt('c.txt', c)
Q = np.loadtxt('Q.txt')
b = np.loadtxt('b.txt')
b = np.array([b]).T
c = np.loadtxt('c.txt')
n=Q.shape[0]

# function for f(x), returns a number
def f(x):
    result = np.matmul(np.matmul(x.T, Q), x) + np.matmul(b.T, x) + c
    return result[0, 0]

# function for df(x)/dx, returns an n*1 vector
def grad_f(x):
    return np.matmul(Q.T + Q, x) + b

def armijo(x, alpha, sigma, beta):
    fx = f(x)
    dk = -grad_f(x)
    grad_f_dk = np.matmul(grad_f(x).T, dk)
    sigma_grad_f_dk = sigma * grad_f_dk[0, 0]
    stop = False
    while not stop:
        lhs = f(x + alpha * x)
        rhs = fx + alpha * sigma_grad_f_dk
        if lhs <= rhs:
            stop = True
        else:
            alpha = alpha * beta
    return alpha

def gradient_descent(x, epsilon, alpha, sigma, beta):
    alpha = armijo(x, alpha, sigma, beta)
    x = x - alpha * grad_f(x)
