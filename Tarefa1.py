# -*- coding: utf-8 -*-

# A = np.matrix("2 1 1 -1 1 ; 0 3 0 1 2 ; 0 0 2 2 -1 ; 0 0 -1 1 2 ; 0 0 0 3 1")

import numpy as np
import math

#------------------# NÃO USAR NO EP #----------------------------
def solvSobr(W,b):
    #Resolve sistema sobredeterminado para medir o erro
    W_ = np.matrix([[0 for j in range(W.shape[1])] for j in range(W.shape[1])], dtype=W.dtype)
    b_ = np.matrix([0 for i in range(W.shape[1])], dtype=b.dtype).T
    for i in range(b_.shape[0]): #percorre linhas
        for j in range(b_.shape[0]): #percorre colunas
            W_[i,j] = (W[:,i].T * W[:,j])[0,0]
        b_[i,0] = (W[:,i].T * b)[0,0]
    return W_.I * b_
#------------------# NÃO USAR NO EP #----------------------------

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
    for r in range(0, m):
        W[i,r], W[j,r] = c*W[i,r] - s*W[j,r], s*W[i,r] + c*W[j,r]


def resolveSobredet(W, b):
    #Aplicação do pseudocódigo dado em 2.3 do enunciado
    n,m = W.shape #n linhas, m colunas
    b_ = b.copy() #cria cópia para não alterar a matriz original
    R = W.copy()
    for k in range(m): #percorrendo colunas
        for j in range(n-1, k, -1): #percorrendo a coluna, de baixo para cima até k+1
            i = j-1
            if W[j,k] != 0:
                c, s = cosSen(R, i, j, k) #acha cos e sen
                rotGivens(R, n, m, i, j, c, s) #aplicação da rotação na marix W
                rotGivens(b_,n, 1, i, j, c, s) #aplicação da rotação no "vetor" b
    #No final M será igual a matriz R, tringular superor, de M original e a matriz b, rotacionada em relação ao b original
    x = np.matrix(m*[0], dtype=W.dtype).T #criação do vetor x como uma matrix m por 1
    for k in range(m-1, -1, -1): #percorre os valores de x
        soma = sum([R[k,j]*x[j,0] for j in range(k+1, m)]) #encontra a somatória para subtrair em b_k
        x[k,0] = (b_[k,0] - soma) / R[k,k] #encontra o x_k
    return x

def testaSobredet():
    #a)
    def temp(k, i):
        if k==i: return 2
        elif abs(k-i)==1: return 1
        elif abs(k-i)>1: return 0
    W = np.matrix([[temp(k,i) for k in range(64)] for i in range(64)], dtype=float)
    b = np.matrix([1+i for i in range(64)], dtype=float).T
    sol = resolveSobredet(W,b)
    #print("Achado:\n", sol)
    #print("Real:\n", Wold.I * bold)
    #pint("Diferenca:\n", (Wold.I * bold) - sol)
    print("Total: ", np.square((W.I * b) - sol).sum())
    #b)
    def temp(k, i):
        if abs(k-i)<=4: return 1 / (i+k+1)
        else: return 0
    W = np.matrix([[temp(k,i) for k in range(17)] for i in range(20)], dtype=float)
    b = np.matrix([1 for i in range(20)], dtype=float).T
    sol = resolveSobredet(W,b)
    print("Total: ", np.square(solvSobr(W,b) - sol).sum())

def resolveSimult(W, A):
    #Aplicação do pseudocódigo dado em 2.4 do enunciado
    n,m = A.shape #n linhas, m colunas de A
    p = W.shape[1] #p colunas de H
    A_ = A.copy() #cria cópia para não alterar a matriz original
    R = W.copy()
    for k in range(p): #percorrendo colunas
        for j in range(n-1, k, -1): #percorrendo a coluna, de baixo para cima até k+1
            i = j-1
            if W[j,k] != 0:
                c, s = cosSen(R, i, j, k) #acha cos e sen
                rotGivens(R, n, p, i, j, c, s) #aplicação da rotação na mariz W
                rotGivens(A_,n, m, i, j, c, s) #aplicação da rotação no matrz A
    #No final M será igual a matriz R, tringular superor, de M original e a matriz A, rotacionada em relação ao A original
    H = np.matrix([[0 for j in range(p)] for i in range(m)], dtype=A.dtype).T #criação do vetor x como uma matrix m por 1
    for k in range(p-1, -1, -1): #percorre os valores de x
        for j in range(m):
            soma = sum([R[k,i]*H[i,j] for i in range(k+1, p)]) #encontra a somatória para subtrair em A_k_j
            H[k,j] = (A_[k,j] - soma) / R[k,k] #encontra o H_k_j
    return H

def testaSimult():
    pass #TODO: remover pass e aplicar a função
        
    def temp(k, i):
        if k==i: return 2
        elif abs(k-i)==1: return 1
        elif abs(k-i)>1: return 0
        
    W = np.matrix([[temp(k,i) for k in range(64)] for i in range(64)], dtype=float)
    A1 = np.matrix([1 for i in range(64)], dtype=float).T
    A2 = np.matrix([i+1 for i in range(64)], dtype=float).T
    A3 = np.matrix([2*(i+1)-1 for i in range(64)], dtype=float).T
    A = np.concatenate((A1,A2,A3),axis=1)
    sol = resolveSimult(W,A)
    #print("Achado:\n", sol)
    #print("Real:\n", Wold.I * bold)
    #pint("Diferenca:\n", (Wold.I * bold) - sol)
    print("Total: ", np.square((W.I * A) - sol).sum())
    #b)
    def temp(k, i):
        if abs(k-i)<=4: return 1 / (i+k+1)
        else: return 0
    W = np.matrix([[temp(k,i) for k in range(17)] for i in range(20)], dtype=float)
    A1 = np.matrix([1 for i in range(20)], dtype=float).T
    A2 = np.matrix([i+1 for i in range(20)], dtype=float).T
    A3 = np.matrix([2*(i+1)-1 for i in range(20)], dtype=float).T
    A = np.concatenate((A1,A2,A3),axis=1)
    sol = resolveSimult(W,A)
    print("Total: ", np.square((W.I * A) - sol).sum())
    
testaSimult()