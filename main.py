# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 18:19:11 2020

@author: David
"""
import create_matrix as cm
import time as t
import numpy as np
import distributions as dist
import solvers 
import approximate as a
import sys
 
d = { 'normal': dist.Normal,  'expradial': dist.ExpRadial, 'weibullradial' : dist.WeibullRadial}
s = { 'lu': solvers.LU, 'factor': solvers.Factorize}


distribution = d[sys.argv[1]]
solver = s[sys.argv[2]]
matrix_size =  int(sys.argv[3])

step_size = 0.1
num_steps = int(1/(2*step_size))

num_iterations = 1000000

actual_det=[]
alpha=[]
Y=[]

for i in range(1, num_steps):
    
    alpha.append(i*step_size)
    
    begin = t.time()
    
    M = cm.create_matrix(matrix_size, alpha[i-1])
    
    actual_det.append(np.linalg.det(M.todense()))
    

    begin = t.time()
    
    Y.append(a.approximate(distribution, solver, M, num_iterations, 1))
    
    my = t.time() - begin

    print('Analytically Computed Determinant: ' ,actual_det[i-1])
    print('Approximated Determinant: ', Y[i-1])
    print('Time to Approximate: ', my)

print(actual_det, Y, sep = '\n')
