# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:49:30 2019

@author: Pedro B. Carvalhaes
"""

# A = np.matrix("2 1 1 -1 1 ; 0 3 0 1 2 ; 0 0 2 2 -1 ; 0 0 -1 1 2 ; 0 0 0 3 1")

import numpy as np
import math

def cosSen(W,i,j,k):
    
    if abs(W[i,k]) > abs(W[j,k]):
        tau = -W[j,k]/W[i,k]
        c=1/math.sqrt(1 + tau**2)
        s=c*tau
    else:
         tau = -W[i,k]/W[j,k]
         s=1/math.sqrt(1 + tau**2)
         c=s*tau
         
    return c,s

def rotGivens(W, n, m ,i, j, c, s):
    for r in range(0, n):
        W[i,r], W[j,r] = c*W[i,r] - s*W[j,r], s*W[i,r] + c*W[j,r]
