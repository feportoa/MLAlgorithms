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