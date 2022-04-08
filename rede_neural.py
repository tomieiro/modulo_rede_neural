import random as rd
import numpy as np

############# CLASSE NEURONIO #############
class Neuronio:
    def __init__(self, index):
        self.index = index
        self.w = [] #Vetor de pesos para o neuronio
        self.y = 0 #Valor de saida para o neuronio
        self.delta = 0 #Valor de delta para o neuronio
        self.erro = 0 # Valor de erro para o neuronio
    
    def ativa(self, entrada):
        aux = self.w[-1] #Utilizando o BIAS como o ultimo peso (+ facilidade de cortes do vetor)
        for i in range(len(self.w)-1): #Note que iteramos todos os pesos com relacao as entradas. Como o BIAS nao tem entrada, o incluimos antes ja que 1*W = W
            aux += entrada[i] * self.w[i] #Produto escalar: (entradas, w)
        return aux

    def tranfere(self, entrada):
        self.y = 1.0/(1.0 + np.exp(-self.ativa(entrada)))
    
    def atualiza_pesos(self, entrada, taxa_aprendizado):
        for i in range(len(entrada)):
            self.w[i] -= taxa_aprendizado * (self.delta * entrada[i])
        self.w[-1] -= taxa_aprendizado * self.delta

############## CLASSE CAMADA ##############
class Camada:
    def __init__(self):
        self.neuronios = [] #Vetor de neuronios para cada 
        self.index = 0
    
    def atualiza_pesos(self, entrada, taxa_aprendizado):
        for i in range(len(self.neuronios)):
            self.neuronios[i].atualiza_pesos(entrada, taxa_aprendizado)

############### CLASSE REDE ################
class Rede:

    def __init__(self, num_neuro_entrada, num_camadas_ocultas, num_neuro_camadas_ocultas, num_neuro_saida):
        self.num_neuro_entrada = num_neuro_entrada
        self.num_camadas_ocultas = num_camadas_ocultas
        self.num_neuro_camadas_ocultas = num_neuro_camadas_ocultas
        self.num_neuro_saida = num_neuro_saida
        self.camadas = []
        
        assert self.num_camadas_ocultas > 0,\
            "Verifique que deve existir pelo menos uma camada oculta!"

        assert str(type(num_neuro_camadas_ocultas)) == "<class 'list'>",\
             "Erro! O terceiro argumento deve ser um vetor de quantidades para cada camada oculta!"
        
        assert len(self.num_neuro_camadas_ocultas) == self.num_camadas_ocultas,\
             "Erro! O vetor de numero de neuronios para cada camada está incorreto!"

        layer = None

        #Criando as camadas ocultas e seus neuronios
        for i in range(num_camadas_ocultas):
            layer = Camada()
            layer.index = i
            for j in range(num_neuro_camadas_ocultas[i]):
                layer.neuronios.append(Neuronio(j))
            self.camadas.append(layer)

        #Criando a camada de saida e seus neuronios
        layer = Camada()
        layer.index = num_camadas_ocultas
        for i in range(num_neuro_saida):
            layer.neuronios.append(Neuronio(i))
        self.camadas.append(layer)

        #Gerando pesos para cada neuronio da primeira camada oculta
        for i in range(num_neuro_camadas_ocultas[0]):
            for j in range(num_neuro_entrada+1):
                self.camadas[0].neuronios[i].w.append(float(rd.randrange(-5,5)/10))

        #Gerando pesos para cada neuronio das camadas ocultas restantes
        for i in range(1,num_camadas_ocultas):
            for j in range(num_neuro_camadas_ocultas[i]):
                for k in range(num_neuro_camadas_ocultas[i-1]+1):
                    self.camadas[i].neuronios[j].w.append(float(rd.randrange(-5,5)/10))
        
        #Gerando pesos para cada neuronio da camada de saida
        for i in range(num_neuro_saida):
            for j in range(num_neuro_camadas_ocultas[-1]+1):
                self.camadas[-1].neuronios[i].w.append(float(rd.randrange(-5,5)/10))
 
    def forward_propagation(self, linha_entrada):
        aux = linha_entrada
        for i in range(len(self.camadas)):
            aux2 = []
            for j in range(len(self.camadas[i].neuronios)):
                self.camadas[i].neuronios[j].tranfere(aux)
                aux2.append(self.camadas[i].neuronios[j].y)
            aux = aux2
        return aux

    def error_back_propagation(self, y_esperado):

        assert len(y_esperado) == len(self.camadas[i].neuronios),\
            "Erro! O numero de rotulos passado nao eh o mesmo que o numero de saidas da rede!"

        for i in reversed(range(len(self.camadas))):
            #iterando para a camada de saida
            if i == (len(self.camadas)-1):
                for j in range(len(self.camadas[-1].neuronios)):
                    self.camadas[-1].neuronios[j].erro = self.camadas[-1].neuronios[j].y - y_esperado[j]
            #iterando para as demais camadas
            else:
                for j in range(len(self.camadas[i].neuronios)):
                    aux = 0.0
                    for k in range(len(self.camadas[i+1].neuronios)):
                        aux += (self.camadas[i+1].neuronios[k].w[j] * self.camadas[i+1].neuronios[k].delta)
                    self.camadas[i].neuronios[j].erro = aux
            #Calculando os deltas
            for j in range(len(self.camadas[i].neuronios)):
                self.camadas[i].neuronios[j].delta = self.camadas[i].neuronios[j].erro * (self.camadas[i].neuronios[j].y * (1.0 - self.camadas[i].neuronios[j].y))

    def atualiza_pesos(self, entrada, taxa_aprendizado):
        nova_entrada = []
        for i in range(len(self.camadas)):
            nova_entrada.clear()
            #Atualizando pesos na primeira camada oculta (que recebe as entradas iniciais)
            if self.camadas[i].index == 0:
                self.camadas[i].atualiza_pesos(entrada, taxa_aprendizado)
            #A nova entrada da camada seguinte serao as saidas da anterior
            for j in range(len(self.camadas[i-1].neuronios)):
                nova_entrada.append(self.camadas[i-1].neuronios[j].y)
            self.camadas[i].atualiza_pesos(nova_entrada, taxa_aprendizado)


