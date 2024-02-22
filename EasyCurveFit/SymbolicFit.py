# Importações
from gplearn.genetic import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sympy import *

# Outras bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split

# Carregar um conjunto de dados de exemplo
df = pd.read_excel('Datasets/01a - Linear Model.xlsx')

# Dividir os dados em características (X) e alvo (y)
X = df.drop('X', axis=1)
y = df['Y']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializar o regressor simbólico
sr = SymbolicRegressor(population_size=1000, generations=20, stopping_criteria=0.01,
                       p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05,
                       p_point_mutation=0.1, max_samples=0.9, verbose=1,
                       parsimony_coefficient=0.01, random_state=0)

# Ajustar o modelo aos dados de treinamento
sr.fit(X_train, y_train)

# Exibir a expressão matemática do melhor modelo encontrado
print(sr._program)

