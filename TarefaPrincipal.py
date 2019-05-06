import numpy as np
import time
from Tarefa1 import *

def main(ndig_treino=100, p=5):
    dez = 10
    #Carregando oa bases de treino
    digitos = dez * [0]
    for i in [1]:
        digitos[i] = np.matrix(np.genfromtxt("dados_mnist/train_dig{}.txt".format(i), usecols=range(ndig_treino))) / 255
    print("Terminou de carregar")
    #Parte 1
    Ws = dez * [0]
    for i in [1]:
        Ws[i] = fatoraMatriz(digitos[i], p)[0]
    print("Terminou de fatorar")
    #Parte 2
    baseTreino = np.matrix(np.genfromtxt("dados_mnist/test_images.txt", usecols=range(10000)))
    print(baseTreino.shape)
    Hs = dez * [0]
    normas = np.matrix([[0 for i in range(baseTreino.shape[1])] for j in range(dez)], dtype=float) #10 linhas, 1000o colunas
    for i in [1]:
        Hs[i] = resolveSimult(Ws[i], baseTreino)
        print(np.square(baseTreino-Ws[i]*Hs[i])[:,:10])
        print(np.square(baseTreino-Ws[i]*Hs[i]).sum(0)[:,:10])
        print(np.sqrt(np.square(baseTreino-Ws[i]*Hs[i]).sum(0))[:,:10])
        normas[i,:] = np.sqrt(np.square(baseTreino-Ws[i]*Hs[i]).sum(0))
    return normas


def bla():
    tempo = time.time()
    relogio = time.clock()
    #print(np.genfromtxt("dados_mnist/train_dig3.txt").shape)
    nan  = main()
    print(time.time()-tempo)
    print(time.clock()-relogio)
    return nan

if __name__ == "__main__": 
    normas = bla()
    print(((normas[1,:]==1).sum() / nor)