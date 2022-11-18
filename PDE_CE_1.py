import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, csc_matrix
from scipy.sparse.linalg import spsolve

def grid(N, ansatz='linear'):
    # N: the number of elements
    # ansatz: linear(default) will generate 2 nodes per element. quadratic(else case) generates 3 nodes per element
    # return: a vector containing the node coordinates
    x_min = 0 #Boundary of the domain
    x_max = 1 #Boundary of the domain
    if (ansatz=='linear'):
        n_o_e = N+1 #Number of the elements in the domain
        h = x_max/N #Distance between grids
        vec = np.zeros(n_o_e)
        for i in range(n_o_e):
            vec[i] = x_min + h*i
        
    else: #Quadratic case
        n_o_e = 2*N + 1
        h = x_max / (2*N)
        vec = np.zeros(n_o_e)
        for i in range(n_o_e):
            vec[i] = x_min + h*i
        
    return vec

def assembleMatrix(lattice, ansatz='linear'):
    # lattice: node vector
    # return: stiffness matrix
    A = np.zeros((len(lattice),len(lattice))) #Stiffness matrix with zeros
    A[0,0] = 1 #Altering the matrice for BCs
    A[len(lattice)-1,len(lattice)-1] = 1 #Altering the matrice for BCs
    if (ansatz=='linear'): #Linear part
        for i in range(1,len(lattice)-1): #Diagonal elements
            for j in range(len(lattice)):       
                if (i==j):
                    h = lattice[i]-lattice[i-1]
                    A[i,j] = 2/h
                else:
                    A[i,j] = 0
                    
        for i in range(1,len(lattice)-1):
            for j in range(len(lattice)):       
                if (j==i+1):
                    h = lattice[i]-lattice[i-1]
                    A[i,j] = -1/h
                    
        for i in range(1,len(lattice)-1): 
            for j in range(len(lattice)):       
                if (j==i-1):
                    h = lattice[i]-lattice[i-1]
                    A[i,j] = -1/h

    else: #Quadratic part
        for i in range(1,len(lattice)-1): #Diagonal elements
            for j in range(len(lattice)):       
                if (i==j) and (i%2==1):
                    h = lattice[i]-lattice[i-1]
                    A[i,j] = 8 / (3*h)

        for i in range(1,len(lattice)-1): #Diagonal elements
            for j in range(len(lattice)):       
                if (i==j) and (i%2==0):
                    h = lattice[i]-lattice[i-1]
                    A[i,j] = 7 / (3*h)

        for i in range(1,len(lattice)-1):
            for j in range(len(lattice)):       
                if (j==i-1) or (j==i+1):
                    h = lattice[i]-lattice[i-1]
                    A[i,j] = -4 / (3*h)

        for i in range(1,len(lattice)-1): 
            for j in range(len(lattice)):       
                if (j==i-2) and (i%2 == 0):
                    h = lattice[i]-lattice[i-1]
                    A[i,j] = 1 / (6*h)

        for i in range(1,len(lattice)-1): 
            for j in range(len(lattice)):       
                if (j==i+2) and (i%2 == 0):
                    h = lattice[i]-lattice[i-1]
                    A[i,j] = 1 / (6*h)

    return A

def rhsConstant(lattice, ansatz='linear'):
    # lattice: node values
    # return: vector of right hand side values
    b = np.zeros(len(lattice)) #rhs matrice with zeros
    b[0] = 0 #BC
    b[len(lattice)-1] = 0 #BC
    h = lattice[2]-lattice[1]
    
    if (ansatz=='linear'):
        for i in range(1,len(lattice)-1):
            b[i] = h
    
    else: #Quadratic case
        for i in range(1,len(lattice)-1):
            if(i%2 == 1):
                b[i] = (4*h) / 3
            if(i%2 == 0):
                b[i] = (2*h)/3

    return b


def solConstant(x):
# x: a real number (or a vector of real numbers), where the analytic solution is computed
# return: a real number (or a vector of real numbers) of the analytic solution for f=1
    X =  -0.5*(x-0.5)**2+1/8
    return X

def FEM1DConstant(N, ansatz='linear'):
    # N: number of elements
    # ansatz: choose between 'linear' or 'quadratic' ansatz functions
    # return: pair (node vector, solution vector)
    
    # Set up the node vector
    if (ansatz=='linear'):
        n_o_e = N+1
        u = np.zeros(n_o_e)
        G = grid(N, ansatz='linear')
        A = assembleMatrix(G,'linear')
        b = rhsConstant(G,'linear')
        u = np.linalg.solve(A,b)
    else:
        n_o_e = 2*N + 1
        u = np.zeros(n_o_e)
        G = grid(N, ansatz='quadratic')
        A = assembleMatrix(G,'quadratic')
        b = rhsConstant(G,'quadratic')
        u = np.linalg.solve(A,b) 

    
    return G, u
    

#Solution of the problem
solution = FEM1DConstant(6,ansatz='linearsdf')
grid_vec = solution[0]
approx_sol  = solution[1]



x_exact = np.linspace(0, 1, num=1000)
y_exact = np.zeros(len(x_exact))
for i in range(len(x_exact)):
    y_exact[i] = solConstant(x_exact[i])

#Plotting the numerical and exact solutions
plt.plot(grid_vec, approx_sol,x_exact, y_exact)
plt.legend(('Numerical Solution','Exact Solution'))
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Exact and Numerical Solutions")
plt.show()