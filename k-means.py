"""K-Means

Definir K (Clusters)
Definir (aleatoriamente) um centroide pra cada cluster
Calcular o centroide mais proximo pra cada ponto
Reposicionar o centroide (media de todos os clusters)
Repetir os dois ultimos passos ate atingir a posicao ideal de cada centroide
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

"""Vamos supor que temos varias lojas e queremos estabelecer um processo logistico eficiente utilizando o K-Means para 
conecta-las com os centros logisticos mais proximos"""

dataset = np.array(
    #Aqui temos a matriz de cada loja com as coordenadas geográficas
    [[-25, -46], #são paulo
     [-22, -43], #rio de janeiro
     [-25, -49], #curitiba
     [-30, -51], #porto alegre
     [-19, -43], #belo horizonte
     [-15, -47], #brasilia
     [-12, -38], #salvador
     [-8, -34], #recife
     [-16, -49], #goiania
     [-3, -60], #manaus
     [-22, -47], #campinas
     [-3, -38], #fortaleza
     [-21, -47], #ribeirão preto
     [-23, -51], #maringa
     [-27, -48], #florianópolis
     [-21, -43], #juiz de fora
     [-1, -48], #belém
     [-10, -67], #rio branco
     [-8, -63] ]) #Porto Velho
plt.scatter(dataset[:,1], dataset[:,0]) # posicionamento dos eixos x e y
plt.xlim(-75, -30) # range do eixo x
plt.ylim(-50, 10) # range do eixo y
plt.savefig("Grade.png") # função que desenha a grade no nosso gráfico

kmeans = KMeans(n_clusters = 3, # numero de clusters
                init = 'k-means++', n_init = 10, # init tem o metodo de inicializacao, n_init inicia o calcula 10 vezes e pega o com menor distancia calculada
                max_iter = 300) # numero máximo de iterações
# Os clusters representam os CENTROS LOGISTICOS

pred_y = kmeans.fit_predict(dataset)
plt.scatter(dataset[:,1], dataset[:,0], c = pred_y) # posicionamento dos eixos x e y
plt.xlim(-75, -30) # range do eixo x
plt.ylim(-50, 10) # range do eixo y
plt.grid() # função que desenha a grade no nosso gráfico
plt.scatter(kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,0], s = 70, c = 'red') # posição de cada centroide no gráfico
plt.savefig("kmeans centros logisticos30.png")
