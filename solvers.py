# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:10:51 2020

@author: David
"""

import scipy.sparse as sparse
import scipy.sparse.linalg as spalg
import scipy.fft as scifft
import numpy as np
class LU:
    def __init__(self, M):
        self.LU = spalg.splu(M)
        
    def solve(self, x):
        y = self.LU.solve(x)
        return y


class Factorize:
    def __init__(self, M):
        self.factorized = spalg.factorized(M)
        
    def solve(self, x):
        y = self.factorized(x)
        return y

class FFT:
    def __init__(self, M):
        self.diag_inv = np.reciprocal(np.real(scifft.fft(M[:,0].todense().T)[0]))

    def solve(self, x):
        v = scifft.fft(x)
        u = self.diag_inv*v
        y = scifft.ifft(u)
        return np.real(y)
         
        
class FFTInv:
    def __init__(self, M):
        diag_inv = sparse.diags(np.reciprocal(np.real(scifft.fft(M[:,0].todense().T)[0]))).todense()
        A = scifft.ifft(diag_inv).T
        self.inv = np.real(scifft.fft(A))

    def solve(self, x):
        y = np.dot(self.inv, x)
        return y

if __name__ == '__main__':
    import create_matrix as cm
    n = 4 
    M = cm.create_matrix(n,0.1)
    solver =  FFTInv(M)
    print(solver.inv)
    print(solver.solve(x))


