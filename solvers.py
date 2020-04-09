# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:10:51 2020

@author: David
"""

import scipy.sparse.linalg as spalg

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
