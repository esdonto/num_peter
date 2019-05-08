import numpy as np
from Tarefa1 import *
import time

def main():
    matriz = np.matrix(np.genfromtxt("bla.csv", delimiter=";"))
    tempo = time.time()
    relogio = time.clock()
    W, H = fatoraMatriz(matriz, 10)
    print(time.time() - tempo)
    print(time.clock()- relogio)

if __name__ == "__main__" : main()