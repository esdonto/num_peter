# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:49:30 2019

@author: Pedro B. Carvalhaes
"""

# A = np.matrix("2 1 1 -1 1 ; 0 3 0 1 2 ; 0 0 2 2 -1 ; 0 0 -1 1 2 ; 0 0 0 3 1")

import numpy as np
import math

def cosSen(W,i,j,k):
    #Fórmulas (3) e (4), na parte 2.2 do enunciado
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
    #Aplicação do pseudocódigo dado em 2.4 do enunciado
    for r in range(0, n):
        W[i,r], W[j,r] = c*W[i,r] - s*W[j,r], s*W[i,r] + c*W[j,r]


def resolveSobredet(W, n, m, b):
    #Aplicação do pseudocódigo dado em 2.3 do enunciado
    for k in range(m): #percorrendo colunas
        for j in range(n-1, k+1, -1): #percorrendo a coluna, de baixo para cima até k+1
            i = j-1
            if W[j,k] != 0:
                c, s = cosSen(W, i, j, k)
                rotGivens(W, n, m, i, j, c, s)
    #No final M será igual a matrix R, tringular superor, de M
    x = np.matrix(m*[0], dtype=W.dtype).T #criação do vetor x como uma matrix m por 1
    soma = 0 #somatoria que é subtraída em b[k,0]
    for k in range(m-1, 0, -1): #percorre os valores de x
        if k<m-1: soma += W[k,k+1] * x[k+1,0] #na primeira iteração soma tem que ser igual a 1
        x[k,0] = (b[k,0] - soma) / W[k,k]
    return x

#TODO: Testar criando a) e b) e chacando resultados fazendo M^-1 * b quando M quadrado