#### RWTH Aachen University PDE lecture coding exercise 1 ###

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, csc_matrix
from scipy.sparse.linalg import spsolve
import pandas as pd
from scipy import interpolate

# Define the basis function type as well as the right hand side equation type
basis_function_type = 'quadratic'          # Options --> linear/quadratic
rhs_equation_type = 'constant'          # Options --> constant/dirac

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
                if (j==i-2 or j==i+2) and (i%2 == 0):
                    h = lattice[i]-lattice[i-1]
                    A[i,j] = 1 / (6*h)

    return A

def rhsConstant(lattice, ansatz='linear', rhs_type='constant'):
    # lattice: node values
    # return: vector of right hand side values
    b = np.zeros(len(lattice)) #rhs matrice with zeros
    b[0] = 0 #BC
    b[len(lattice)-1] = 0 #BC
    h = lattice[2]-lattice[1]
    # For f=1
    if rhs_type == 'constant':
        if (ansatz=='linear'):
            for i in range(1,len(lattice)-1):
                b[i] = h
    
        else: #Quadratic case
            for i in range(1,len(lattice)-1):
                if(i%2 == 1):
                    b[i] = (4*h) / 3
                if(i%2 == 0):
                    b[i] = (2*h)/3
    
    # For f = 2*dirac(0.5)
    else:
        if ansatz == 'linear' or ansatz == 'quadratic':
            for i in range(1,len(lattice)-1):
                if round(i*h,2) != 0.5:
                    b[i] = 0
                else:
                    b[i] = 2
    return b


def solConstant(x, rhs_type='constant'):
# x: a real number (or a vector of real numbers), where the analytic solution is computed
# return: a real number (or a vector of real numbers) of the analytic solution for f=1 or f = dirac(0.5)
    if rhs_type == 'constant':
        X =  -0.5*(x-0.5)**2+1/8
    else:
        if x <= 0.5:
            X = x
        else:
            X = 1 - x
    return X

def FEM1DConstant(N, ansatz='linear', rhs_type='constant'):
    # N: number of elements
    # ansatz: choose between 'linear' or 'quadratic' ansatz functions
    # return: pair (node vector, solution vector)
    
    # Set up the node vector
    if (ansatz=='linear'):
        n_o_e = N+1
        u = np.zeros(n_o_e)
        G = grid(N, ansatz='linear')
        A = assembleMatrix(G,'linear')
        b = rhsConstant(G, ansatz, rhs_type)
        u = np.linalg.solve(A,b)
    else:
        n_o_e = 2*N + 1
        u = np.zeros(n_o_e)
        G = grid(N, ansatz='quadratic')
        A = assembleMatrix(G,'quadratic')
        b = rhsConstant(G, ansatz, rhs_type)
        u = np.linalg.solve(A,b) 

    
    return G, u
    
# Define the method for interpolation of curve equation from approximate solution vector
def interpolationFunction(x, xk, uk):
    # Inputs: x - vector for the points at interpolation equation; xk - input vector of the approx. solution;
    # uk - vector of the approx. solutions at points xk
    # Using scipy's interpolate function
    f_interpolate = interpolate.interp1d(xk,uk)
    ui = f_interpolate(x)
    # Return the vector with the interpoated values
    return ui

# Define the convergence behaivour between the exact solution and numerical solutions
def convergenceBehaviour(x_input, ansatz, rhs_type):
    # Define a data frame to hold the norm value for the error and the grid size
    df_convergence = pd.DataFrame(columns=['Grid Size','Error'])
    # Define the vector for true solution
    u = np.zeros(len(x_input))
    # Get the true solution vector
    for j in range(len(x_input)):
        u[j] = solConstant(x_input[j], rhs_type)
    # Create a for loop to get the different solitions to the grid sizes
    for i in range(2,40):
        grid_size = 1/i
        # Get the FEM Solution for the current grid size
        fem_soln = FEM1DConstant(i, ansatz, rhs_type)
        # Get the xk values
        xk = fem_soln[0]
        # Get the numerical solution
        uh = fem_soln[1]
        # Get the interpolated value from the interpolation function for the approximated solution
        ui = interpolationFunction(x_input, xk, uh)
        # Get the error values
        error = u - ui
        # Get the L2 error norm
        error_norm = (np.sum(np.power(error,2)))**0.5
        # Add the error into the data frame
        df_convergence = df_convergence.append({'Grid Size':grid_size, 'Error':error_norm}, ignore_index=True)
    # Plot the error graph
    plt.plot(df_convergence['Grid Size'], df_convergence['Error'])
    plt.xlabel('$log(h)$')
    plt.ylabel(r'$log(\left \| u-u_h \right \|_{L^2})$')
    plot_title = 'Loglog PLot - ' + ansatz.capitalize() + ' Basis, ' + rhs_type.capitalize() + ' RHS'
    plt.title(plot_title)
    plt.grid()
    plt.show()
    

#Solution of the problem
solution = FEM1DConstant(18,basis_function_type, rhs_equation_type)
grid_vec = solution[0]
approx_sol  = solution[1]



x_exact = np.linspace(0, 1, num=1000)
y_exact = np.zeros(len(x_exact))
for i in range(len(x_exact)):
    y_exact[i] = solConstant(x_exact[i], rhs_equation_type)


#Plotting the numerical and exact solutions
plt.plot(grid_vec, approx_sol, x_exact, y_exact)
plt.legend(('Numerical Solution','Exact Solution'))
plt.xlabel("x")
plt.ylabel("u(x)")
plot_title = 'Exact Vs. Numerical Solutions - ' + basis_function_type.capitalize() + ' Basis, ' + rhs_equation_type.capitalize() + ' RHS'
plt.title(plot_title)
plt.grid()
plt.show()

convergenceBehaviour(x_exact, basis_function_type, rhs_equation_type)
