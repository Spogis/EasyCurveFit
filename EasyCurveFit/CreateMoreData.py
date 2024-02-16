import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Carrega os dados do arquivo Excel
caminho_do_arquivo = '../Datasets/10 - Stress Strain Curve.xlsx'
df = pd.read_excel(caminho_do_arquivo)

# Supõe que os dados estão nas colunas 'x' e 'y'
x = df['x'].to_numpy()
y = df['y'].to_numpy()

# Gera 5000 pontos igualmente espaçados entre o mínimo e o máximo de x
x_novos = np.linspace(x.min(), x.max(), 10000)

# Cria a função de interpolação
funcao_interp = interp1d(x, y, kind='cubic')  # 'cubic' para interpolação cúbica; pode ajustar conforme necessário

# Usa a função de interpolação para obter os novos valores de y
y_novos = funcao_interp(x_novos)

# (Opcional) Plota os dados originais e os dados interpolados
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label='Dados Originais')
plt.plot(x_novos, y_novos, '-', label='Dados Interpolados')
plt.legend()
plt.show()

# Cria um DataFrame com os novos pontos e salva em um novo arquivo Excel
df_novo = pd.DataFrame({'x': x_novos, 'y': y_novos})
df_novo.to_excel('../Datasets/dados_interpolados.xlsx', index=False)
