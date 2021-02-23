import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


#Faz a leitura do arquivo CSV usando pandas.
dataset = pd.read_csv(r'04_dados_exercicio.csv')  #raw

# Cria uma base contendo as variáveis independentes e uma base contendo a variável dependente.
features = dataset.iloc[:, :-1].values
classe = dataset.iloc[:, -1].values
print(features)
print(classe)

# Substitui dados faltantes pela média da respectiva variável
imputer = SimpleImputer(missing_values=np.nan, strategy="mean") #construimos o obj, nan e o valor a ser substituido e mean é a media
imputer.fit(features[:, 2:4]) #aonde temos os dados faltantes
features[:, 2:4] = imputer.transform(features[:, 2:4])
print(features)

# Codifica todas as variáveis categóricas independentes com One Hot Encoding.
columnTransformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
features = np.array(columnTransformer.fit_transform(features))
print(features)

# Codica a variável dependente com Codicação por Rótulo
labelEncoder = LabelEncoder()
classe = labelEncoder.fit_transform(classe)
print('==========features===========')
print(features)
print('==========classe===========')
print(classe)

# Separa a base em duas partes: uma para treinamento e outra para testes.
# Use 85% das instâncias para o treinamento.
features_treinamento, features_teste, classe_treinamento, classe_teste = train_test_split(features, classe, test_size = 0.15, random_state=1)
print('==========features_treinamento===========')
print(features_treinamento)
print('==========features_teste===========')
print(features_teste)
print('==========classe_treinamento===========')
print(classe_treinamento)
print('==========classe_teste===========')
print(classe_teste)


# Normaliza as variáveis temperatura e humidade usando padronização.
standardScaler = StandardScaler()
features_treinamento[:, 2:4] = standardScaler.fit_transform(features_treinamento[:, 2:4])
features_teste[:, 2:4] = standardScaler.transform(features_teste[:, 2:4])
print('==========features_treinamento===========')
print(features_treinamento)
print('==========features_teste===========')
print(features_teste)





