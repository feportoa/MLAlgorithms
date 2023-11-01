"""Random Forest

Metodo estatistico

Treinar varias arvores de decisao
Se treinarmos varias arvores com o mesmo dataset, todas vao ter a mesma predicao
Se o dataset tiver 1 erro, todas as arvores vao ter o mesmo erro
Solucao: Bootstraping!
Pegar pequenas amostras do dataset original, fazer copias (Bootstrap samples)
Juntar diversas destas pequenas amostras em um outro dataset (Bagging / Bootstrap aggregating)

Random Forests sao treinadas com diferentes conjuntos de dados
Feature Randomness - selecionar features aleatorias

Todo esse processo de aleatorizacao eh feito para minizar as semelhancas com o dataset original, diminuindo a chance
de erros

"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Primeiro, vamos criar um dataset com a relação y = x
x = pd.DataFrame({'val': np.linspace(0, 10, 10)})
y = x.val
plt.plot(x, y)
plt.title('Saída correta') # Linha que define a precisao, quanto mais proximo melhor
plt.savefig('Saida correta.png')

# Criamos o modelo de regressão com 50 árvores
rf = RandomForestRegressor(50)
rf.fit(x, y)

# Predições da random forest
x_pred = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = rf.predict(x_pred)

# predições das duas primeiras árvores do nosso modelo
y_pred0 = rf.estimators_[0].predict(x_pred)
y_pred1 = rf.estimators_[1].predict(x_pred)
y_pred2 = rf.estimators_[2].predict(x_pred)

plt.plot(x_pred, y_pred)
plt.title('Random Forest')
plt.savefig('Random Forest.png')

plt.plot(x_pred, y_pred0, label='Decision Tree 0')
plt.title('Decision Tree 0')
plt.savefig('Decision tree 0.png')

plt.plot(x_pred, y_pred1, label='Decision Tree 1')
plt.title('Decision Tree 1')
plt.savefig('Decision Tree 1.png')

plt.plot(x_pred, y_pred2, label='Decision Tree 2')
plt.title('Decision Tree 2')
plt.savefig('Decision Tree 2.png')