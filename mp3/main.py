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
# method=sys.argv[8]



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
def update_c(c, method='multiply', hx_norm=None, last_hx_norm=None):
    if method == 'const':
        newc = c
    elif method == 'add':
        newc = c + 0.5
    elif method == 'multiply':
        newc = c * 1.1
    else:
        if not hx_norm or not last_hx_norm:
            raise ValueError('For method 4, need h(xk) and h(x(k-1))!')
        if hx_norm > 0.9 * last_hx_norm:
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
        # print(i)
        last_x = x
        x, _, _ = minimizer.gradient_descent(x, lambda_k, c_k)
        lambda_k = lambda_k + c_k * minimizer.h(x)
        if method == 'multiply_with_cond':
            c_k = update_c(c_k, method=method, hx_norm=minimizer.norm_h(x), last_hx_norm=minimizer.norm_h(last_x))
        else:
            c_k = update_c(c_k,method=method)
        x_result.append(x)
        c_k_result.append(c_k)
        i = i + 1
    
    x_error=[]
    if(np.linalg.norm(x) !=0):
        for vec in x_result:
            x_error.append( np.linalg.norm(vec-x)/np.linalg.norm(x) )

    return x,i,x_error

x0 = np.zeros((Q.shape[0],1))
lambda0 = np.zeros((A.shape[0],1))

# x, num_iter,x_error = method_of_multiplier(epsilon=0.00001, x=x0, c_k=1,method="const",lambda_k=lambda0)
# print('Method 1: solution x:', x.flatten(), '\n, number of iterations:', num_iter)
# plt.plot(x_error,'go', label='method 1',markersize=5)

x, num_iter,x_error = method_of_multiplier(epsilon=0.00001, x=x0, c_k=1,method="add",lambda_k=lambda0)
print('Method 2: solution x:', x.flatten(), '\n, number of iterations:', num_iter)
plt.plot(x_error,'rx', label='add',markersize=5)

x, num_iter,x_error = method_of_multiplier(epsilon=0.00001, x=x0, c_k=1,method="multiply",lambda_k=lambda0)
print('Method 3: solution x:', x.flatten(), '\n, number of iterations:', num_iter)
plt.plot(x_error,'b*', label='multiply',markersize=5)

# x, num_iter,x_error = method_of_multiplier(epsilon=0.00001, x=x0, c_k=1,method="m4",lambda_k=lambda0)
# print('Method 4: solution x:', x.flatten(), '\n, number of iterations:', num_iter)
# plt.plot(x_error,'y1', label='method 3',markersize=5)


plt.legend()
plt.ylabel('relative error')
plt.xlabel('iteration number')
plt.show()
linear_constraint = LinearConstraint(A, b.flatten(), b.flatten())

    


def rosen(x,Q):
    c=Q.dot(x)
    return np.dot(x,c)

res = optimize.minimize(rosen, x0.flatten(),Q, method='SLSQP', constraints=linear_constraint)

print("scipy optimizer solution:",res.x)

