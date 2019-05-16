import numpy as np
import matplotlib.pyplot as plt
import time
from Tarefas12 import *

def fazPredicao(ndig_treino=100, p=5):
    dez = 10
    #Carregando oa bases de treino
    digitos = dez * [0]
    for i in range(dez):
        digitos[i] = np.genfromtxt("dados_mnist/train_dig{}.txt".format(i), usecols=range(ndig_treino)) / 255
    #print("Terminou de carregar")
    #Parte 1
    Ws = dez * [0]
    for i in range(dez):
        Ws[i] = fatoraMatriz(digitos[i], p)[0]
        np.savetxt("dados_gerados/Ws/{}/W{}-n{}p{}.txt".format(i, i, ndig_treino, p), Ws[i], delimiter=",")
    #print("Terminou de fatorar")
    #Parte 2
    dataTeste = np.genfromtxt("dados_mnist/test_images.txt", usecols=range(10000)) / 255
    Hs = dez * [0]
    normas = np.array([[0 for i in range(dataTeste.shape[1])] for j in range(dez)]) #10 linhas, 10000 colunas
    for i in range(dez):
        Hs[i] = resolveSimult(Ws[i], dataTeste)
        normas[i,:] = np.sqrt(np.square(dataTeste-Ws[i]@Hs[i]).sum(0))
    np.savetxt("dados_gerados/normas/normas-n{}p{}.txt".format(ndig_treino,p), normas, delimiter=",")
    predicao = np.zeros(10000)
    for i in range(10000):
        maxNorma = normas[0,i]
        for j in range(1, dez):
            if maxNorma > normas[j,i]:
                predicao[i] = j
                maxNorma = normas[j,i]
    np.savetxt("dados_gerados/predicao/predicao-n{}p{}.txt".format(ndig_treino,p), predicao, delimiter=",")

def calculaErro(ndig_treino=100, p=5):
    predicao = np.genfromtxt("dados_gerados/predicao/predicao-n{}p{}.txt".format(ndig_treino,p), delimiter=",")
    ids = np.genfromtxt("dados_mnist/test_index.txt")

    quantidadeDeAcertos = (predicao == ids).sum()

    quantidadeDeDigitos = 10 * [0]
    acertosDeDigitos = 10 * [0]
    for i in range(10): 
        quantidadeDeDigitos[i] = (ids == i).sum()
        acertosDeDigitos[i] = (predicao[ids==i] == ids[ids==i]).sum()

    return quantidadeDeAcertos, quantidadeDeDigitos, acertosDeDigitos

def imprimeErros(ndig_treinos=[100,1000,4000], ps=[5,10,15]):
    for ndig_treino in ndig_treinos:
        for p in ps:
            quantidadeDeAcertos, quantidadeDeDigitos, acertosDeDigitos = calculaErro(ndig_treino, p)
            print("ndig_treino = {} , p = {}: acertos = {}%".format(ndig_treino, p, quantidadeDeAcertos/100))
            for i in range(10):
                print("Digito {}: acertos = {} / {} = {}%".format(i, acertosDeDigitos[i], quantidadeDeDigitos[i], 100*acertosDeDigitos[i]/quantidadeDeDigitos[i]))
            print()

def plotaW(ndig_treino=100, p=5, digito=0):
    W = np.genfromtxt("dados_gerados/Ws/{}/W{}-n{}p{}.txt".format(digito, digito, ndig_treino, p), delimiter=",")
    for i in range(p):
        plt.subplot(p//5, 5, i+1)
        plt.imshow(W[:,i].reshape(28,28)).set_cmap('Greys')
        plt.xticks([])
        plt.yticks([])
    plt.suptitle("Componentes do digito {} para ndig_treino={} e p={}".format(digito, ndig_treino, p), fontsize=30)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1, top=0.9)
    plt.show()

def plotaDigito(dig):
    plt.imshow(dig.reshape(28, 28)).set_cmap("Greys")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

def permutaPredicoes(ndig_treinos=[100,1000,4000], ps=[5,10,15]):
    for ndig_treino in ndig_treinos:
        for p in ps:
            t = time.time()
            fazPredicao(ndig_treino, p)
            print("ndig_treino={} p={} => {}s".format(ndig_treino, p, time.time()-t))
