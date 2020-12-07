import numpy as np
from gd_armijo import GD_Armijo
import matplotlib.pyplot as plt 
import sys
import scipy.optimize as optimize

from scipy.optimize import LinearConstraint

alpha0=float(sys.argv[4])
beta=float(sys.argv[5])
sigma=float(sys.argv[6])
epsilon=float(sys.argv[7])
method=sys.argv[8]



Q=np.loadtxt(sys.argv[1])
A=np.loadtxt(sys.argv[2])
b=np.loadtxt(sys.argv[3])

b=b[:,np.newaxis]


# m = 2, n = 3
# Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# A = np.array([[1, 2, 3], [4, 5, 6]])
# b = np.array([[1], [1]])
# epsilon=0.00001
# alpha0=0.001
# sigma=0.1
# beta=0.1

# initialize the gradient descent optimizer for L(x, c, lambda)
minimizer = GD_Armijo(Q, A, b, epsilon, alpha0, sigma, beta)

# given ck, calculate c(k+1)
# method: whether we are using method 1, 2, 3, 4 in Hint
# 1: const, 2: add, 3: multiply, 4: multiply_with_condition
def update_c(c, method='multiply', hx=None, last_hx=None):
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
def method_of_multiplier(epsilon, x, c_k,method,lambda_k):
    i = 0
    x_result=[]
    c_k_result=[]
    while minimizer.norm_h(x) >= epsilon:
        x, _, _ = minimizer.gradient_descent(x, lambda_k, c_k)
        lambda_k = lambda_k + c_k * minimizer.h(x)
        c_k = update_c(c_k,method=method)
        x_result.append(x)
        c_k_result.append(c_k)
        i = i + 1
    
    x_error=[]
    if(np.linalg.norm(x) !=0):
        for vec in x_result:
            x_error.append( np.linalg.norm(vec-x)/np.linalg.norm(x) )
    plt.plot(c_k_result,x_error)
    plt.ylabel('relative error')
    plt.xlabel('c_k')
    plt.show()
    return x,i

x0 = np.zeros((Q.shape[0],1))
lambda0 = np.zeros((A.shape[0],1))

x, num_iter = method_of_multiplier(epsilon=0.00001, x=x0, c_k=1,method=method,lambda_k=lambda0)
print('solution x:', x.flatten(), '\n, number of iterations:', num_iter)

linear_constraint = LinearConstraint(A, b.flatten(), b.flatten())

    


def rosen(x,Q):
    c=Q.dot(x)
    return np.dot(x,c)

res = optimize.minimize(rosen, x0.flatten(),Q, method='SLSQP', constraints=linear_constraint)

print("scipy optimizer solution:",res.x)



# Todo: 1. add plotting for part3; 2. check with scipy optimize