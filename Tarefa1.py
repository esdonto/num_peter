# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:49:30 2019

@author: Pedro B. Carvalhaes
"""
import numpy as np
import math

def cosSen(w,i,j,k):
    
    if abs(w[i,k]) > abs(w[j,k]):
        tau = -w[j,k]/w[i,k]
        c=1/math.sqrt(1 + tau**2)
        s=c*tau
    else:
         tau = -w[i,k]/w[j,k]
         s=1/math.sqrt(1 + tau**2)
         c=s*tau
         
    return c,s

def rotGivens(W, n, m ,i, j, c, s):
    
    for r in range(0, n):
        print(W[i,r], W[j,r], end='')
        aux = c*W[i,r] - s*W[j,r]
        W[j,r] = s*W[i,r] + c*W[j,r]
        W[i,r] = aux
        #W[i,r], W[j,r] = c*W[i,r] - s*W[j,r], s*W[i,r] + c*W[j,r]
        print("  ola  ", W[i,r], W[j,r])
