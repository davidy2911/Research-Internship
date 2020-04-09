# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:43:15 2020

@author: David
"""

import numpy as np
#import time as t

def approximate(distribution, solver, M, n, debug = 0):
    if debug == 1 : print('Approximation begun')
    size = M.shape[0]
    
    dist = distribution(size)
    
    if debug == 1: print('Distribution generated')
    
    L = solver(M)
    
    if debug == 1: print('Solver generated')
    
    Y=[]
    
    for i in range(n):
           
           if debug == 1 and i%100000 == 0: print('Iteration ', i, ' of ', n)
           
           x = dist.generate()
           
           y = L.solve(x[0])
           
           Y.append(dist.f(x, y))
    
    return np.mean(Y)
