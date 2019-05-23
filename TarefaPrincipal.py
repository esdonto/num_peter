import numpy as np
import matplotlib.pyplot as plt
import time
from Tarefas12 import *


#---------------------------------------------------------Predições-------------------------------------------------------------------

def fazPredicao(ndig_treino=100, p=5):
    '''Recebe a quantidade ndig_treino de amostras de cada digito que deve carregar para montar a matriz W 28² por p e acha a norma para cada digito para a base base de teste.
    Carrega os digitos da pasta dados_mnist, usa a função fatoraMatriz da Tarefas12 para achar W e salva esse W's,
    depois usa a função resolveSimult da Tarefas12 para encontrar a norma de cada imagem da base de teste em relação aos W's gerados de cada tipo de imageme salva essas normas'''
    #Carregando oa bases de treino
    digitos = 10 * [0]
    for i in range(10):
        digitos[i] = np.genfromtxt("dados_mnist/train_dig{}.txt".format(i), usecols=range(ndig_treino)) / 255 #Divide por 255 para normalizar cada valor da matriz, não é necessário ja que dentro de fatoraMatriz suas colunas já são normalizadas
    Ws = 10 * [0] #Um W para cada digito possível
    for i in range(10):
        Ws[i] = fatoraMatriz(digitos[i], p)[0] #Encontra o W de cada digito possível
        np.savetxt("dados_gerados/Ws/{}/W{}-n{}p{}.txt".format(i, i, ndig_treino, p), Ws[i], delimiter=",") #Salva esse W
    #Fase de encontrar os Ws terminada, agora precisso carregar a base de teste e achar as normas
    dataTeste = np.genfromtxt("dados_mnist/test_images.txt", usecols=range(10000)) #Carrega a base de teste
    Hs = 10 * [0] #Um H para cada digito
    normas = np.array([[0 for i in range(dataTeste.shape[1])] for j in range(10)]) #10 linhas, 10000 colunas
    for i in range(10): #itera cada digito
        Hs[i] = resolveSimult(Ws[i], dataTeste) #Encontra o H para a base de teste em relação a cada digito possível
        normas[i,:] = np.sqrt(np.square(dataTeste-Ws[i]@Hs[i]).sum(0)) #Calcula a norma do H escontrado
    np.savetxt("dados_gerados/normas/normas-n{}p{}.txt".format(ndig_treino,p), normas, delimiter=",") #Salva a norma encontrada
    #Fase de encontrar as normas terminada terminada, agora preciso fazer a predição
    predicao = np.zeros(10000) #Uma predição por imagem da base de teste
    for i in range(10000):
        minNorma = normas[0,i] #Inicia a mínima norma como a norma do para o dígito 0
        for j in range(1, 10):
            if minNorma > normas[j,i]: #Caso a nomrma de outro didito for meonr, se atualiza a norma mínima e troca o digito que se acha que é para o de norma menor
                predicao[i] = j
                minNorma = normas[j,i]
    np.savetxt("dados_gerados/predicao/predicao-n{}p{}.txt".format(ndig_treino,p), predicao, delimiter=",") #Salva a predição

def permutaPredicoes(ndig_treinos=[100,1000,4000], ps=[5,10,15]):
    '''Faz todas as combinações possíveis de ndig_treinos e p pedidas no enunciado e imprime o tempo que cada predição tomou'''
    for ndig_treino in ndig_treinos:
        for p in ps:
            t = time.time()
            fazPredicao(ndig_treino, p)
            print("ndig_treino={} p={} => {}s".format(ndig_treino, p, time.time()-t))


#---------------------------------------------------------Acerto-------------------------------------------------------------------

def calculaAcerto(ndig_treino=100, p=5):
    '''Dados a quantidade ndig_treino de cada digito usada para gerar cada W 28² por p uasadas na predição,
    devolve a quanto acertou no total, a quantidade de cada digito na base de teste e quanto se acertou de cada digito'''

    predicao = np.genfromtxt("dados_gerados/predicao/predicao-n{}p{}.txt".format(ndig_treino,p), delimiter=",") #Carrega apredição feita com ndig_treino e p
    ids = np.genfromtxt("dados_mnist/test_index.txt") #Carrega o valor certo de cada digito da base de teste

    quantidadeDeDigitos = 10 * [0]
    acertosDeDigitos = 10 * [0]
    for i in range(10): 
        quantidadeDeDigitos[i] = (ids == i).sum() #Quantidade de valores de cada digito na base de teste que são iguais a i
        acertosDeDigitos[i] = (predicao[ids==i] == i).sum() #Quantidade de valores que deveriam ser igual a i e obteve uma previsão certa

    quantidadeDeAcertos = np.sum(acertosDeDigitos) #Soma a quantidade total de acertos

    return quantidadeDeAcertos, quantidadeDeDigitos, acertosDeDigitos

def imprimeAcertos(ndig_treinos=[100,1000,4000], ps=[5,10,15]):
    '''Imprime a quantidade de acertos de cada digito de totais para cada um dos ndig_treino's e p's dadoss'''

    for ndig_treino in ndig_treinos: #itera cada um dos ndig_treino's
        for p in ps: #itera cada um dos p's
            quantidadeDeAcertos, quantidadeDeDigitos, acertosDeDigitos = calculaAcerto(ndig_treino, p) #Calcula a quantidade de acertos
            print("ndig_treino = {} , p = {}: acertos = {}%".format(ndig_treino, p, quantidadeDeAcertos/100)) #Imprime total de acertos
            for i in range(10):
                print("Digito {}: acertos = {} / {} = {}%".format(i, acertosDeDigitos[i], quantidadeDeDigitos[i], 100*acertosDeDigitos[i]/quantidadeDeDigitos[i])) #Imprime a quantidade de acertos de cada digito
            print()


#---------------------------------------------------------Plots-------------------------------------------------------------------

def plotaW(ndig_treino=4000, p=15, digito=4):
    '''Dados a quantidade ndig_treino de cada digito usada para gerar o W 28² por p do digito selecionado, plota os componentes desse W'''
    W = np.genfromtxt("dados_gerados/Ws/{}/W{}-n{}p{}.txt".format(digito, digito, ndig_treino, p), delimiter=",") #Carrega o W
    for i in range(p):
        plt.subplot(p//5, 5, i+1)
        plt.imshow(W[:,i].reshape(28,28)).set_cmap('Greys')
        plt.xticks([])
        plt.yticks([])
    plt.suptitle("Componentes do digito {} para ndig_treino={} e p={}".format(digito, ndig_treino, p), fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1, top=0.9)
    plt.show()

def plotaImagem(img):
    '''Dado a matriz com 28² componentes, plota a imagem que essa matriz codifica'''
    plt.imshow(img.reshape(28, 28)).set_cmap("Greys")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()


