import numpy as np
from scipy.optimize import minimize

# function for f(x), returns a number
# x: a list with length 3
# w: a list with length 3
def f(w, x):
    result = 1/(1-x[0]-x[1])**2 + 1/(1-x[0]-x[2])**2 - (w[0]*np.log(x[0])+w[1]*np.log(x[1])+w[2]*np.log(x[2]))
    return result

# function for df(x)/dx, returns an n*1 vector
def grad_f(w, x):
    df = np.array([0, 0, 0])
    term1 = 2/(1-x[0]-x[1])**3
    term2 = 2/(1-x[0]-x[2])**3
    # dg/dx1, dg/dx2, dg/dx3
    df[0] = term1 + term2 - w[0]/x[0]
    df[1] = term1 - w[1]/x[1]
    df[2] = term2 - w[2]/x[2]
    return df


# Armijo's rule for finding alpha given x_k and other parameters
def armijo(w, x, alpha0, sigma, beta):
    alpha = alpha0
    fx = f(w, x)
    dk = -grad_f(w, x)
    grad_f_dk = np.dot(grad_f(w, x), dk)
    sigma_grad_f_dk = sigma * grad_f_dk
    stop = False
    while not stop:
        lhs = f(w, x + alpha * dk)
        rhs = fx + alpha * sigma_grad_f_dk
        if lhs <= rhs:
            stop = True
        else:
            alpha = alpha * beta
    return alpha

def gradient_descent(w, x, epsilon, alpha0, sigma, beta):
    i = 0
    while abs(np.linalg.norm(grad_f(w, x))) >= epsilon:
        alpha = armijo(w, x, alpha0, sigma, beta)
        x = x - alpha * grad_f(w, x)
        i = i + 1
        # print(x)
    return x, f(w, x), i


def part1():
    w_list = [[1, 1, 1], [1, 2, 3], [2, 2, 2]]
    x0_list = [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25],[0.25, 0.25, 0.25]]
    part_num = ['a', 'b', 'c']
    for i in range(3):
        # armijo's rule
        x_star1, f_x_star1, iter_num = gradient_descent(w_list[i], x0_list[i], epsilon=0.00001, alpha0=0.001, sigma=0.001, beta=0.2)
        print('Part', part_num[i], ':')
        print('Armijo rule:')
        print('x*:', x_star1, ', f(x*):', f_x_star1, 'number of iterations:', iter_num)

        def f1(x):
            result = 1 / (1 - x[0] - x[1] - x[2]) ** 2 - (
                        w_list[i][0] * np.log(x[0]) + w_list[i][1] * np.log(x[1]) + w_list[i][2] * np.log(x[2]))
            return result
        cons = ({'type': 'ineq',
                 'fun': lambda x: np.array([-x[0]-x[1]-x[2]+1])})
        # scipy.optimize
        # print('91')
        result = minimize(f1, np.array(x0_list[i]), method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
        print('scipy.optimize:')
        print('x*:', result.x, ', f(x*):', f1(result.x))

# main function
part1()