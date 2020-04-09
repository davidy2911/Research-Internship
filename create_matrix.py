# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 18:23:48 2020

@author: David
""",
import numpy as np
import scipy.sparse as sparse

def create_matrix(n, a):
    diags = [-a]*(n-1)
    M = sparse.diags([np.ones(n),diags,diags,diags, diags],[0,-1,1,1-n,n-1],(n,n),'csc')
    return M

