# -*- coding: utf-8 -*-

import numpy as np
import math

def cosSen(W,i,j,k):
    '''Dadas as linhas i e j e a coluna k da matriz W, encontra o seno e cosseno de W[i,k] e W[j,k], segundo as fórmulas (3) e (4) da parte 2.2 do enunciado''' 
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
    '''Aplica a rotação de Givens nas linhas i e j da matriz W, dados os seno e cosseno dessa rotação, seguindo o pseudocódigo dado em 2.4 do enunciado'''
    W[i,:], W[j,:] = c*W[i,:] - s*W[j,:], s*W[i,:] + c*W[j,:]


def resolveSobredet(W, b):
    '''Resolve o sistema sobredeterminado W*x = b, retornando o vetor x que minimiza o erro quadrado médio da equação, aplicando o pseudocódigo apresentado na parte 2.3 do enunciado'''
    n,m = W.shape #n linhas, m colunas
    b_ = b.copy() #cria cópia para não alterar a matriz original
    R = W.copy() #cria cópia para não alterar a matriz original
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
    print("a) E = ", np.sqrt(np.square(W@sol - b).sum()))
    print("Erro do MMQ = ", np.sqrt(np.square((W.T@W)@sol - (W.T@b)).sum()))
    #b)
    def temp(k, i):
        if abs(k-i)<=4: return 1 / (i+k+1)
        else: return 0
    W = np.array([[temp(k,i) for k in range(17)] for i in range(20)], dtype=float)
    b = np.array([[i+1] for i in range(20)], dtype=float)
    sol = resolveSobredet(W,b)
    print("b) E = ", np.sqrt(np.square(W@sol - b).sum()))
    print("Erro do MMQ = ", np.sqrt(np.square((W.T@W)@sol - (W.T@b)).sum()))

def resolveSimult(W, A):
    '''Resolve os sistemas sobredeterminados W*x = b simultaneamente, sendo x cada couna de H e b cada coluna de A,
    retornando os vetores x's que minimizam o erro quadrado médio das equações, sendo cada uma das colunas de H um desses x's, 
    aplicando o pseudocódigo apresentado na parte 2.4 do enunciado'''
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
            if R[k,k]: H[k,j] = (A_[k,j] - soma) / R[k,k]
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
    print("c) E = ", np.sqrt(np.square(W@sol - A).sum()))
    print("Erro do MMQ = ", np.sqrt(np.square((W.T@W)@sol - (W.T@A)).sum()))

    #d)
    def temp(k, i):
        if abs(k-i)<=4: return 1 / (i+k+1)
        else: return 0
    W = np.array([[temp(k,i) for k in range(17)] for i in range(20)], dtype=float)
    A = np.array([[1 for i in range(20)], [i+1 for i in range(20)], [2*(i+1)-1 for i in range(20)]], dtype=float).T
    sol = resolveSimult(W,A)
    print("d) E = ", np.sqrt(np.square(W@sol - A).sum()))
    print("Erro do MMQ = ", np.sqrt(np.square((W.T@W)@sol - (W.T@A)).sum()))
 
def fatoraMatriz(A,p):
    '''Fatora uma matriz A n por m não negativa em uma matriz W n por p e H p por m, tal que tenta minimizar o módulo de erro A-W*H.
    Implementação do pseudocódigo dado em na parte 3 do enunciado'''
    n,m = A.shape #A n por m, W n por p e H p por m
    # = A.copy() #cria cópia para não alterar a matrix A original
    itmax = 100 #número de iterações para chegar a condição de saída
    e = 1e-5 #valor da diferença entre dois erros consecutivod máximo para chegar na condição de saída
    
    W = np.random.rand(n,p) #Cria matriz W com todos os valores aleatórios
    
    it=0 #número de iterações
    E = 0
    deltaE = 1 #diferença entre os erros de 2 iteração consecutivas, começando maior do que e para não ser pego na condiçaõ de saída

    while it<itmax and deltaE>e:
        s = np.sqrt(np.square(W).sum(0)) #faz o quadrado de todos os valores de W, soma as colunas e depois a raiz quadrada de cada somatória
        W /= s #subtrai cada coluna de W por cada item de s
                
        H = resolveSimult(W,A)
        H[H<0] = 0 #Faz com que H seja matriz positiva

        Wtrsp = resolveSimult(H.T, A.T)
        W = Wtrsp.T #Acha transposta
        W[W<0] = 0 #Faz com que W seja matriz negativa
        
        Eantigo, E = E, np.square(A-W@H).sum() #Acha novo valor para E, fazendo
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
    Es = 100*[0]
    for i in range(100):
        W_, H_ = fatoraMatriz(A, p)
        Es[i] = np.square(A - W_@H_).sum()
    print("E da fatoração da matriz no enunciado: ", np.mean(Es))

if __name__ == "__main__" :
    print("Tarefa 1:")
    testaSobredet()
    testaSimult()
    print("\nTarefa 2:")
    testaFatora()

#A = np.matrix("68 78 39 146 18 59 139; 42 134 105 123 50 79 88; 88 97 109 131 38 73 88; 64 54 28 82 32 78 44; 28 62 53 52 14 37 39; 186 187 96 225 84 231 158", dtype=float)
#temp = fatoraMatriz(A, 5)
#print(np.dot(temp[0], temp[1]))