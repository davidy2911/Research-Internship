# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 18:19:11 2020

@author: David
"""

"""First I import some standard scientific modules for  timing, for dealing with arrays
 and for reading input off command line """

import time as t          
import numpy as np          
import sys

"""Then I import the modules I have written for this project that handle creating the 
large sparse matrix, generating samples from the distributions, solving the linear system, 
and combining these to do the actaul monte carlo simualtion and approximation"""

import create_matrix as cm
import distributions as dist
import solvers as sol
import approximate as app

""""I create a dictionary of possible inputs from the command line and the functions and 
classes they respsond to in the distributions and solvers modules"""
 
d = { 'normal': dist.Normal,  'expradial': dist.ExpRadial, 'weibullradial' : dist.WeibullRadial}
s = { 'lu': sol.LU, 'factor': sol.Factorize, 'fft': sol.FFT, 'fft_inv': sol.FFTInv}

"""These iterative generators are then assigned to variables"""

distribution = d[sys.argv[1]]
solver = s[sys.argv[2]]
matrix_size =  int(sys.argv[3])

"""The step size to move the alpha and the number of iterations are hard coded in as 
I rarley need to change them """

step_size = 0.1
num_steps = int(1/(2*step_size))

num_iterations = 1000000

"""I declare the empty lists that values will be appended too"""

actual_det=[]
alpha=[]
approximated_det=[]

for i in range(1, num_steps):
    
    alpha.append(i*step_size)
    
    begin = t.time()
    
    M = cm.create_matrix(matrix_size, alpha[i-1])
    
    actual_det.append(np.linalg.det(M.todense()))

    begin = t.time()
    
    approximated_det.append(app.approximate(distribution, solver, M, num_iterations, 1))
    
    my = t.time() - begin

    print('Analytically Computed Determinant: ' ,actual_det[i-1])
    print('Approximated Determinant: ', approximated_det[i-1])
    print('Time to Approximate: ', my)

print(actual_det, approximated_det, sep = '\n')
