import numpy as np
import sys
# Choose n, then generate random values for Q (positive definite),b, and c
n=int(sys.argv[1])
m=int(sys.argv[2])
Q=np.random.random((n,n))
Q=Q.dot(Q.T)+0.1*np.identity(n)
A=np.random.random((m,n))
b=np.random.random(m)
np.savetxt('Q.txt', Q)
np.savetxt('A.txt', A)
np.savetxt('b.txt', b)
