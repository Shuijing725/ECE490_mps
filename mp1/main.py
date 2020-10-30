import numpy as np
# Choose n, then generate random values for Q (positive definite),b, and c
n=5
Q=np.random.random((n,n))
Q=Q.dot(Q.T)+0.1*np.identity(n)
b=np.random.random(n)
c=np.random.random(1)
np.savetxt('Q.txt', Q)
np.savetxt('b.txt', b)
np.savetxt('c.txt', c)
#print(Q)
f = open("Q.txt", "r")
#print(f.readline())

import autograd.numpy as np
from autograd import grad

#Now we will define the objective function Obj=X(T).Q.X+b(T).X+c

def function(x):
    return np.dot(np.dot(np.transpose(x), Q), x) + np.dot(np.transpose(b), x) + c

#Now we will define the parameters for Armijo's Rule: Sigma, Alpha and Beta
Sigma=0.1
Alpha=1
Beta=0.5
K=0
Error = 1.1e-6

x = np.ones(n)
#print(x)
g = grad(function)(x)
#print(function(x))
#print(g)

from numpy import linalg as LA

norm_g = LA.norm(g)

#use Error = 1.e-6 and use the steepest descent direction -g

#while norm_g > Error:
d=-1*g
print ("direction of the gradient=",d)
print("norm of gradient =", norm_g)
#g_T=(np.transpose(g))
new_x=function(x+Alpha*d)
print ("new_x=", new_x)
print ("function x=",function(x))
print ("new_x - old_x=",new_x-function(x))
print("RHS=",(function(x) - Alpha*Sigma*norm_g*norm_g))
while norm_g > Error:
    while new_x > (function(x) - Alpha*Sigma*norm_g*norm_g):
        Alpha=Alpha*Beta
        new_x=function(x+Alpha*d)
        print("Alpha=",Alpha)
        print("K=",K)
        print("RHS=", (function(x) - Alpha * Sigma * norm_g * norm_g))
        print("LHS=", new_x)
        K = K + 1

    x = x + Alpha * d
    g = grad(function)(x)
    norm_g = LA.norm(g)


    print("Final K=",K)
    print("Final Alpha=",Alpha)
    print("gradient norm=", norm_g)
    print("x*=", x)
    print("F(x*)=", new_x)
    break



