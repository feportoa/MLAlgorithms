"""Apriori

Regra de Associacao
Metodo de explorar relacoes entre itens em conjuntos de dados

Calculo de Support:
Medida que indica a proporcao de X em D (Database)
Supp(X) = (#X in D)/(#D)

Fazemos o suporte com 1 item de cada vez, depois com 2 itens, 3, 4, ate nao sobrar mais nada que ultrapasse o threshold
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori

# Montando dataset
colunas = ['id', 'cerveja', 'fralda', 'chiclete', 'refrigerante', 'salgadinho']
dataset = [[1, 1, 1, 1, 1, 0],
           [2, 1, 1, 0, 0, 0],
           [3, 1, 1, 1, 0, 1],
           [4, 1, 1, 0, 1, 1],
           [5, 0, 1, 0, 1, 0],
           [6, 0, 1, 0, 0, 0],
           [7, 0, 1, 0, 0, 0],
           [8, 0, 0, 0, 1, 1],
           [9, 0, 0, 0, 1, 1]]

# Supp(X) = 4/9

# DataFrame
dataFrame = pd.DataFrame(dataset, columns=colunas)

class Apriori:
    """Classe que contem todos os passos do Apriori"""
    threshold = 0.5 # Numero considerado "limite" para o dataset
    dataFrame = None # Sintaxe esquisita do caralho, todo mundo usa null

    def __init__(self, dataFrame, threshold=None, transform_bool=False):
        """Explicando o construtor do Apriori

        :parametro pandas.DataFrame dataFrame: dataset das transacoes (1 ou 0).
        :parametro float threshold: coloca o limitador (threshold) para suporte_min
        :retorno: Instancia do Apriori
        :tipoRetorno: Apriori
        """

        self._validate_df(dataFrame)

        self.dataFrame = dataFrame
        if threshold is not None:
            self.threshold = threshold

        if transform_bool:
            self._transform_bool()

    def _validate_df(self, df=None):
        """Valida existencia do DataFrame

        :parametro pandas.DataFrame df: dataset das transacoes (1 ou 0)
        :retorno:
        :tipoRetorno: void
        """

        if df is None:
            raise Exception("df deve ser um pandas.DataFrame valido.")

    def _transform_bool(self):
        """Transforma dataset (1 ou 0) para True ou False

        :retorno:
        :tipoRetorno: void
        """

        for colunas in self.dataFrame.columns:
            self.dataFrame[colunas] = self.dataFrame[colunas].apply(lambda x: True if x == 1 else False) # Troca os valores do dataset (1 e 0) para True e False

    def _apriori(self, use_colnames=False, max_len=None, count=True):
        """Chama funcao do mlxtend.frequent_patterns

        :parametro bool use_colnames: Marca pra usar nomes de colunas no DataFrame final
        :parametro int max_len: comprimento maximo dos itemsets gerados
        :parametro bool count: Marca pra contar o comprimento dos itemsets
        :retorno: DataFrame apriori
        :tipoRetorno: pandas.DataFrame
        """

        apriori_dataFrame = apriori(
            self.dataFrame,
            min_support=self.threshold, # Limitador
            use_colnames=use_colnames,
            max_len=max_len
        )

        if count:
            apriori_dataFrame['length'] = apriori_dataFrame['itemsets'].apply(lambda x: len(x))

        return apriori_dataFrame

    def run(self, use_colnames=False, max_len=None, count=True):
        """Funcao runner do Apriori

        :parametro bool use_colnames: Marca pra usar o nome das colunas no DataFrame final
        :parametro int max_len: Comprimento maximo dos itemsets gerados
        :parametro bool count: Marca pra contar ou nao o comprimento dos itemsets
        :retorno: DataFrame apriori
        :tipoRetorno: pandas.DataFrame
        """

        return self._apriori(
            use_colnames=use_colnames,
            max_len=max_len,
            count=count
        )

    def filter(self, apriori_df, length, threshold):
        """Filtra o DataFrame apriori por comprimento e threshold

        :parametro pandas.DataFrame apriori_df: DataFrame do apriori
        :parametro int length: Comprimento dos itemsets requeridos
        :parametro float threshold: limitador minimo requerido
        :retorno: DataFrame do apriori filtrado
        :tipoRetorno: pandas.DataFrame
        """

        if 'length' not in apriori_df.columns:
            raise Exception("Apriori nao tem tamanho. Por favor, execute o apriori com count=True.")

        return apriori_df[ (apriori_df['length'] == length) & (apriori_df['support'] >= threshold) ]

# Rodando o Apriori!

if 'id' in dataFrame.columns: del dataFrame['id'] # id nao tem papel relevante, entao deletamos

apriori_runner = Apriori(dataFrame, threshold=0.4, transform_bool=True)
apriori_df = apriori_runner.run(use_colnames=True)
print(apriori_df)
