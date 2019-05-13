# -*- coding: utf-8 -*-

# A = np.matrix("2 1 1 -1 1 ; 0 3 0 1 2 ; 0 0 2 2 -1 ; 0 0 -1 1 2 ; 0 0 0 3 1")

import numpy as np
import math

#------------------# NÃO USAR NO EP #----------------------------
def solvSobr(W,b):
    #Resolve sistema sobredeterminado para medir o erro
    W_ = np.array([[0 for j in range(W.shape[1])] for j in range(W.shape[1])], dtype=W.dtype)
    b_ = np.array([[0 for i in range(b.shape[1])] for k in range(W.shape[1])], dtype=b.dtype)
    for i in range(W.shape[1]): #percorre linhas
        for j in range(W.shape[1]): #percorre colunas
            W_[i,j] = (W[:,i].T @ W[:,j])
    for k in range(b.shape[1]):
        for i in range(b_.shape[0]):
            b_[i,k] = (W[:,i].T @ b[:,k])
    return np.linalg.inv(W_) @ b_
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
    W[i,:], W[j,:] = c*W[i,:] - s*W[j,:], s*W[i,:] + c*W[j,:]


def resolveSobredet(W, b):
    #Aplicação do pseudocódigo dado em 2.3 do enunciado
    n,m = W.shape #n linhas, m colunas
    b_ = b.copy() #cria cópia para não alterar a matriz original
    R = W.copy()
    for k in range(m): #percorrendo colunas
        for j in range(n-1, k, -1): #percorrendo a coluna, de baixo para cima até k+1
            i = j-1
            if R[j,k] != 0:
                c, s = cosSen(R, i, j, k) #acha cos e sen
                rotGivens(R, n, m, i, j, c, s) #aplicação da rotação na marix W
                rotGivens(b_,n, 1, i, j, c, s) #aplicação da rotação no "vetor" b
    #No final M será igual a matriz R, tringular superor, de M original e a matriz b, rotacionada em relação ao b original
    x = np.ones((m,1), dtype=W.dtype) #criação do vetor x como uma matrix m por 1
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
    W = np.array([[temp(k,i) for k in range(64)] for i in range(64)], dtype=float)
    b = np.array([[1 for i in range(64)]], dtype=float).T
    sol = resolveSobredet(W,b)
    #print("Achado:\n", sol)
    #print("Real:\n", Wold.I * bold)
    #pint("Diferenca:\n", (Wold.I * bold) - sol)
    #print("Total: ", np.square((np.linalg.inv(W) @ b) - sol).sum())
    print("a) Erro: ", np.sqrt(np.square(W@sol - b).sum()))
    #b)
    def temp(k, i):
        if abs(k-i)<=4: return 1 / (i+k+1)
        else: return 0
    W = np.array([[temp(k,i) for k in range(17)] for i in range(20)], dtype=float)
    b = np.array([[i+1 for i in range(20)]], dtype=float).T
    sol = resolveSobredet(W,b)
    #print("Total: ", np.square(solvSobr(W,b) - sol).sum())
    print("b) Erro: ", np.sqrt(np.square(W@sol - b).sum()))
testaSobredet()

def resolveSimult(W, A):
    #Aplicação do pseudocódigo dado em 2.4 do enunciado
    n,m = A.shape #n linhas, m colunas de A
    p = W.shape[1] #p colunas de H
    A_ = A.copy() #cria cópia para não alterar a matriz original
    R = W.copy()
    for k in range(p): #percorrendo colunas
        for j in range(n-1, k, -1): #percorrendo a coluna, de baixo para cima até k+1
            i = j-1
            if R[j,k] != 0:
                c, s = cosSen(R, i, j, k) #acha cos e sen
                rotGivens(R, n, p, i, j, c, s) #aplicação da rotação na mariz W
                rotGivens(A_,n, m, i, j, c, s) #aplicação da rotação no matrz A
    #No final M será igual a matriz R, tringular superor, de M original e a matriz A, rotacionada em relação ao A original
    H = np.zeros((p,m), dtype=A.dtype) #criação do vetor x como uma matrix m por 1
    for k in range(p-1, -1, -1): #percorre os valores de x
        for j in range(m):
            soma = np.sum(R[k, k+1:] @ H[k+1:, j]) #encontra a somatória para subtrair em A_k_j
            if R[k,k] : H[k,j] = (A_[k,j] - soma) / R[k,k]
            else: H[k,j] = (A_[k,j] - soma) / 1e-15 #encontra o H_k_j
    return H

def testaSimult():  
    #c)      
    def temp(k, i):
        if k==i: return 2
        elif abs(k-i)==1: return 1
        elif abs(k-i)>1: return 0
        
    W = np.array([[temp(k,i) for k in range(64)] for i in range(64)], dtype=float)
    A = np.array([[1 for i in range(64)], [i+1 for i in range(64)], [2*(i+1)-1 for i in range(64)]], dtype=float).T
    sol = resolveSimult(W,A)
    #print("Total: ", np.square((np.linalg.inv(W) @ A) - sol).sum())
    print("c) Erro: ", np.sqrt(np.square(W@sol - A).sum()))
    #d)
    def temp(k, i):
        if abs(k-i)<=4: return 1 / (i+k+1)
        else: return 0
    W = np.array([[temp(k,i) for k in range(17)] for i in range(20)], dtype=float)
    A = np.array([[1 for i in range(20)], [i+1 for i in range(20)], [2*(i+1)-1 for i in range(20)]], dtype=float).T
    sol = resolveSimult(W,A)
    #print("Total: ", np.square(solvSobr(W,A) - sol).sum())
    print("d) Erro: ", np.sqrt(np.square(W@sol - A).sum()))
    
testaSimult()
def fatoraMatriz(A,p):
    #Implementação do pseudocódigo dado em na parte 3 do enunciado
    n,m = A.shape #A n por m, W n por p e H p por m
    A_ = A.copy() #cria cópia para não alterar a matrix A original
    itmax = 100 #número de iterações para chegar a condição de saída
    e = 1e-5 #valor da diferença entre dois erros consecutivod máximo para chegar na condição de saída
    
    W = np.ones((n,p), dtype=A.dtype) #Cria matriz W com todos os valores iguais a 1
    
    it=0 #número de iterações
    E = 0
    deltaE = 1 #diferença entre os erros de 2 iteração consecutivas, começando maior do que e para não ser pego na condiçaõ de saída

    while it<itmax and deltaE>e:
        s = np.sqrt(np.square(W).sum(0)) #faz o quadrado de todos os valores de W, soma as colunas e depois a raiz quadrada de cada somatória

        W /= s
                
        H = resolveSimult(W,A_)

        H[H<0] = 0

        Wtrsp = resolveSimult(H.T, A_.T)
        W = Wtrsp.T

        W[W<0] = 0
        
        Eantigo, E = E, np.square(A_-W@H).sum()

        deltaE = abs(E - Eantigo)
        it += 1
        #print(E, it)
        
    return W,H

        
def testaFatora():
    #Testa o erro da fatoração criada pelo método de mínimos quadrados alternados
    A = np.array([[3, 6, 0],[5, 0, 10],[4, 8, 0]], dtype=float) / 10
    W = np.array([[3, 0],[0, 5],[4, 0]], dtype=float) / 5
    H = np.array([[1, 2, 0],[1, 0, 2]], dtype=float) / 2
    m, n, p = 3, 3, 2

    W_, H_ = fatoraMatriz(A, p)
    #print("\nW\n", W, "\nW_\n", W_, "\nW-W_\n", W-W_)
    #print("\nH\n", H, "\nH_\n", H_, "\nH-H_\n",  H-H_)
    #print("\nA\n", A, "\nA_\n", W_@H_, "\nA-A_\n", A - W_@H_)
    print("Fatoração: ", np.square(A - W_@H_).sum())

testaFatora()

#A = np.matrix("68 78 39 146 18 59 139; 42 134 105 123 50 79 88; 88 97 109 131 38 73 88; 64 54 28 82 32 78 44; 28 62 53 52 14 37 39; 186 187 96 225 84 231 158", dtype=float)
#temp = fatoraMatriz(A, 5)
#print(np.dot(temp[0], temp[1]))