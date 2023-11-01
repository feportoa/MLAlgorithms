"""Entropia

Medida de desorganização
+ entropia - ganho de informacao
- entropia + ganho de informacao

ENTROPIA:
        n
H(X) = -Σ p(Xi)Log2p(Xi)
       i=1

Σ (Sigma) Representa iteração na matemática
Tradução para ADSês

p = [0.2, 0.3, 0.1, 0.15, 0.25] # Esse eh o nosso p(Xi), ele representa probabilidade e eh um valor que ja estaria calculado em outra parte do codigo
n = len(p) # quantidade dos nossos dados, nesse caso, o comprimento (length) de p, que é 5

res = 0 # Esse vai ser o resultado, comeca como 0 pq nao temos dados ainda

for(i = 0; i <= n-1; i++){ # o "i" representa o indice, tanto no sigma, quanto no for P.S.: Nao escrevi com sintaxe de python pq nao estamos acostumados com ela
    H = p[i] * math.log(p[i]) # H representa H(X), que é a entropia
    res += H; # Res soma os valores anteriores com a entropia atual, no fim, tera o resultado completo da entropia final
}

entropiaFinal = res * (-1) # O resultado sempre sera negativo, portanto multiplicamos por -1 e assim fazemos com que o res seja positivo

GANHO DE INFORMACAO:
Capacidade de uma feature separar os registros conforme suas classes

Calculado comparando a entropia atual (pai) com uma entropia que seria obtida apos uma nova ramificacao (filho)
Para calcular essa formula precisaremos do peso do filho:

peso(FILHO) = n de elementos no node do filho ÷ n de elementos no node do pai

Agora vamos para a formula de ganho de informacao:

                n
ganho = H(PAI) -Σ peso(FILHOi) * H(FILHOi)
               i=1

Calculamos o ganho para cada uma das features

Nossa ramificacao na arvore sera feita com base na coluna que teve o maior ganho de informacao, e repetimos esse
processo recursivamente pra cada lado da ramificacao, so parando qdo o ganho for 0

O objetivo eh ter a menor entropia possivel (vulgo maximo de informacao), para isso, a arvore tem que crescer bastante

Se a arvore crescer demais, pode causar OVERFITTING, ai tem que podar
"""

import numpy as np # Calculos
import pandas as pd # Analise e manipulacao de dados
import matplotlib.pyplot as plot # Visualizacao de dados / Graficos
import seaborn as sns # Tambem graficos!
import pydotplus # Representa GRAFOS e nao GRAFICOS!

dataset = pd.read_csv('wine.data', header = None)

dataset.columns = ['label',
                   'alcohol',
                   'malic_acid',
                   'ash',
                   'alcalinity_of_ash',
                   'magnesium',
                   'total_phenols',
                   'flavanoids',
                   'nonflavanoid_phenols',
                   'proanthocyanins',
                   'color_intensity',
                   'hue',
                   'OD280/OD315',
                   'proline']

print(dataset.head()) # Visualizar as 5 primeiras linhas da tabela de vinhos

# Vamos normalizar os dados para o sistema nao considerar uma variavel mais ou menos importante

# Divisao treino-teste

from sklearn.model_selection import train_test_split # Importando treino teste

x = dataset.values[:, 1:] # [Linha, Coluna] o primeiro ':' significa "Pegue todas as linhas",
                          # o '1:' significa "Pegue todas as colunas exceto a primeira" (A primeira coluna eh a origem
                          # do vinho. Nao vamos usar aqui)

y = dataset.values[:, 0] # a primeira coluna do dataset indica a origem do vinho

# x = dados de treino / teste | y = respostas do treino / teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
# test_size eh a % de dados usados no teste, nesse caso 20% dos dados sao teste e os outros 80% sao treino
# random_state eh o estado que gera numeros aleatorios, se deixar no 0, a divisao dos dados deve ser consistente

#Feature Scaling

from sklearn.preprocessing import StandardScaler # Importando padronizador

scaler = StandardScaler() # Inicializando o bixinho
scaler.fit(x_train) # Calcula a media e padrao pra ser usado no scaling

# Padronizacao:
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Bora pro treino:
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Treina uma arvore com um determinado tamanho
def train_model(height):
    model = DecisionTreeClassifier(criterion = 'entropy', max_depth = height, random_state = 0)
    model.fit(x_train, y_train)
    return model

for height in range(1, 21): # 1-20
    model = train_model(height)
    y_pred = model.predict(x_test)

    print('--------------------------------------------------------------\n')
    print(f'Altura - {height}\n')
    print("Precisão: " + str(accuracy_score(y_test, y_pred)))

# Exportando a arvore e as informações como imagem:


from IPython.display import Image
from sklearn.tree import export_graphviz

model = train_model(3) # vamos importar o modelo com maior precisao, que foi a arvore de altura 3


feature_names = ['alcohol',
                 'malic_acid',
                 'ash',
                 'alcalinity_of_ash',
                 'magnesium',
                 'total_phenols',
                 'flavanoids',
                 'nonflavanoid_phenols',
                 'proanthocyanins',
                 'color_intensity',
                 'hue',
                 'OD280/OD315',
                 'proline']

classes_names = ['%.f' % i for i in model.classes_]

dot_data = export_graphviz(model, filled=True, feature_names=feature_names, class_names=classes_names, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)

Image(graph.create_png())
graph.write_png("tree.png")
Image('tree.png')