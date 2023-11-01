"""Regressao Linear

Correlacao:
Variavel dependente y altera de acordo com a variavel independente x
Exemplo: o peso(y) de uma pessoa varia de acordo com a sua altura(x)

Temos o coeficiente da correlacao de Pearson que indica força (de -1 até 1) e direçao da reta ('-1 \' '0 -' '1 /')
Essas sao correlacoes perfeitas, mas isso nem sempre (na verdade quase nunca) acontece, entao medimos as correlacoes
como fracas, moderadas e fortes

Regressao:
A regressao mede a relacao atraves de uma equacao em vez de so medir a forca da correlacao
Pode ser usada para descrever o quanto o peso aumenta baseando-se na altura e prever pesos para alturas especificas

y = b0 + b1*X

y: variavel dependente (Valor previsto)
X: variavel independente
b0: coeficiente que corta o eixo y
b1: coeficiente que define a inclinacao da reta

(Pode-se adicionar o erro padrao a equacao adicionando um "+ e" ao final)

Para definir o b1:
b1 = Σ((Xi - média(x)) * (yi - média(y))) / Σ((Xi - média(x))^2)

media(x): x eh o valor total, ou seja, eh a soma dos Xi

b0 = média(y) - b1 * média(x)

coeficiente de determinacao (R^2)
Mostra o quao proximos os dados estao da linha de regressao (Mais perto = melhor), esse valor vai de 0 a 1
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# definindo listas X e y
X = np.array([1.47, 1.5, 1.52, 1.55, 1.57, 1.63, 1.65, 1.68, 1.7, 1.73, 1.78, 1.8, 1.83]).reshape(-1, 1)
y = np.array([52.21, 53.12, 57.2, 58.6, 58.57, 59.93, 62, 63.11, 65.4, 66.28, 68.1, 72.19, 80.24])

model = LinearRegression()

model.fit(X, y) # treinando o modelo

# imprimindo o coeficiente que corta o eixo y
intercept = model.intercept_
print(f'Coeficiente de interceptação: {intercept:.4f}')
# imprimindo o coeficiente de inclinação da reta
slope = model.coef_
print(f'Coeficiente de inclinação:    {slope.round(4)}')

print(f"y = {intercept} + {slope} * X")

# Prevendo com os dados de X
y_pred_eq = intercept + slope * X
print(y_pred_eq.tolist())

# Prevendo utilizando dados do X
y_pred_model = model.predict(X)
print(y_pred_model)

# definindo a área de plotagem
plt.figure(figsize=(8,6)) # 800x600

# definindo o gráfico
plt.scatter(X, y,  color='blue')
plt.plot(X, model.predict(X), color='red', linewidth=2)

# Tickando X e y
plt.xticks(np.arange(1.40, 1.95, 0.1))
plt.yticks([50, 60, 70, 80])

# inserindo os rótulos dos eixos
plt.xlabel("Altura (m)")
plt.ylabel("Peso (kg)")

plt.title("Demonstração de Regressão Linear", fontweight="bold", size=15)
plt.savefig("GraficoLR.png")