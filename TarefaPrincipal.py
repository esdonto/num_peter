import numpy as np
import time
from Tarefa1 import *

def main(ndig_treino=100, p=5):
    dez = 10
    #Carregando oa bases de treino
    digitos = dez * [0]
    for i in range(dez):
        digitos[i] = np.matrix(np.genfromtxt("dados_mnist/train_dig{}.txt".format(i), usecols=range(ndig_treino))) / 255
    print("Terminou de carregar")
    #Parte 1
    Ws = dez * [0]
    for i in range(dez):
        Ws[i] = fatoraMatriz(digitos[i], p)[0]
        np.savetxt("Ws/n4000p15{}.csv".format(i), Ws[i], delimiter=",")
    print("Terminou de fatorar")
    #Parte 2
    baseTreino = np.matrix(np.genfromtxt("dados_mnist/test_images.txt", usecols=range(10000)))
    print(baseTreino.shape)
    Hs = dez * [0]
    normas = np.matrix([[0 for i in range(baseTreino.shape[1])] for j in range(dez)]) #10 linhas, 1000o colunas
    for i in range(dez):
        Hs[i] = resolveSimult(Ws[i], baseTreino)
        normas[i,:] = np.sqrt(np.square(baseTreino-Ws[i]*Hs[i]).sum(0))
    np.savetxt("normas.csv", normas, delimiter=",")
    return normas


def bla():
    tempo = time.time()
    relogio = time.clock()
    #print(np.genfromtxt("dados_mnist/train_dig3.txt").shape)
    nan  = main()
    print(time.time()-tempo)
    print(time.clock()-relogio)
    return nan