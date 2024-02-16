
# Substitua 'your_data_file.xlsx' pelo caminho do seu arquivo Excel
input_filename = '../Datasets/09 - RDP.xlsx'  # Atualize para o caminho do seu arquivo

# Lendo os dados do arquivo Excel
df = pd.read_excel(input_filename)

# Aplicando o filtro Gaussiano
sigma = 2  # Ajuste conforme necessário para controlar o grau de suavização
y_filtered_gaussian = gaussian_filter(df['y'].values, sigma)

# Criando um DataFrame com os dados filtrados (usando y_filtered_gaussian como exemplo)
df_filtered = pd.DataFrame({'x': df['x'].values, 'y_filtered': y_filtered})