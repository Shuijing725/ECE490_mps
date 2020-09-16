import numpy as np
from scipy.optimize import minimize

# Choose n, then generate random values for Q (positive definite),b, and c
# n=5
# Q=np.random.random((n,n))
# Q=Q.dot(Q.T)+0.1*np.identity(n)
# b=np.random.random(n)
# c=np.random.random(1)
# np.savetxt('Q.txt', Q)
# np.savetxt('b.txt', b)
# np.savetxt('c.txt', c)

# load from txt
Q = np.loadtxt('Q.txt')
b = np.loadtxt('b.txt')
b = np.array([b]).T
c = np.loadtxt('c.txt')

# hw2 problem
# Q = np.array([[3, 0], [0, 4]])
# b = np.array([[0], [0]])
# c = np.array([[0]])

# find n automatically
n=Q.shape[0]

# function for f(x), returns a number
# x: a n*1 2d array
def f(x):
    result = np.matmul(np.matmul(x.T, Q), x) + np.matmul(b.T, x) + c
    return result[0, 0]

# re-define f(x) for scipy.optimize because result is a 1d array somehow
def f1(x):
    result = np.matmul(np.matmul(x.T, Q), x) + np.matmul(b.T, x) + c
    return result[0]

# function for df(x)/dx, returns an n*1 vector
def grad_f(x):
    return np.matmul(Q.T + Q, x) + b

# Armijo's rule for finding alpha given x_k and other parameters
def armijo(x, alpha0, sigma, beta):
    alpha = alpha0
    fx = f(x)
    dk = -grad_f(x)
    grad_f_dk = np.matmul(grad_f(x).T, dk)
    sigma_grad_f_dk = sigma * grad_f_dk[0, 0]
    stop = False
    while not stop:
        lhs = f(x + alpha * dk)
        rhs = fx + alpha * sigma_grad_f_dk
        if lhs <= rhs:
            stop = True
        else:
            alpha = alpha * beta
    return alpha

def gradient_descent(x, epsilon, alpha0, sigma, beta):
    i = 0
    while abs(np.linalg.norm(grad_f(x))) >= epsilon:
        alpha = armijo(x, alpha0, sigma, beta)
        x = x - alpha * grad_f(x)
        i = i + 1
        # print(x)
    return x, f(x), i


# part 1
# initial guess
x_guess = np.array([[0.], [0.], [0.], [0.], [0.]])
# x_guess = np.array([[1.], [0.]])
x_star1, f_x_star1, iter_num = gradient_descent(x_guess, epsilon=0.00001, alpha0=1, sigma=0.001, beta=0.2)
print('Part 1:')
print('x*:', x_star1, ', f(x*):', f_x_star1, 'number of iterations:', iter_num)

# part 2
# grad_f(x*) = (Q^T + Q)x* + b = 0 => x* = -(Q^T+Q)^(-1) * b
x_star2 = - np.matmul(np.linalg.inv(Q.T + Q), b)
f_x_star2 = f(x_star2)
print('Part 2:')
print('x*:', x_star2, ', f(x*):', f_x_star2)

# part 3
result = minimize(f1, x_guess, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
print('Part 3:')
print('x*:', result.x, ', f(x*):', f1(result.x))
