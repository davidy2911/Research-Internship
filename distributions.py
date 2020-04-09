# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:10:16 2020

@author: David
"""

import numpy as np
import math as m


class Normal:
    def __init__(self, n):
        self.n = n
        
    def generate(self):
        x = np.random.standard_normal(self.n)
        return [x]
    
    def f(self, x , y):
        exponent = np.dot(y.T, y) - np.dot(x[0].T,x[0])
        
        return m.exp(-exponent/2)

class ExpRadial:
    def __init__(self, n):
        self.n = n
        
    def generate(self):
        radius = np.random.standard_exponential()
        
        direction = np.random.standard_normal(self.n)
        norm  = np.linalg.norm(direction)
    
        x = (direction/norm)*radius
    
        return [x, radius] 
    
    def f(self, x, y):
        y_norm = np.linalg.norm(y)
        
        exponent = y_norm + x[1]
        
        frac = x[1]/y_norm
        
        power = m.pow(frac, self.n-1)
        
        return power*m.exp(exponent)
    

class WeibullRadial:
    def __init__(self, n):
        self.n = n
        
    def generate(self):
        radius_sqrt = np.random.standard_exponential()
        radius = radius_sqrt**2
        
        direction = np.random.standard_normal(self.n)
        norm  = np.linalg.norm(direction)
    
        x = (direction/norm)*radius
    
        return [x, radius_sqrt]

    
    def f(self, x, y):
        y_norm_sqrt = m.sqrt(np.linalg.norm(y))
        
        exponent = -y_norm_sqrt + x[1]
        
        frac = x[1]/y_norm_sqrt
        
        power = m.pow(frac, 2*self.n-1)
        
        return power*m.exp(exponent)    