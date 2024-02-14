import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sympy import symbols, sympify, lambdify
from sklearn.metrics import r2_score

from EasyCurveFit.CleanData import *

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def limpar_nome_arquivo(nome_arquivo):
    caracteres_invalidos = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in caracteres_invalidos:
        nome_arquivo = nome_arquivo.replace(char, '_')
    return nome_arquivo

# 1. Ler um arquivo Excel
def ler_excel(caminho):
    return pd.read_excel(caminho)

# 2. Solicita ao usuário a equação para ajustar a curva
def solicita_equacao(equacao):
    return equacao.split('=')[1].strip()  # Retorna somente o lado direito da equação

def ParametersData(equation_input):
    equacao = solicita_equacao(equation_input)
    parametros = sorted(list(set([str(p) for p in sympify(equacao).free_symbols if str(p) != 'x'])))
    return parametros

# 3. Função para ajuste de curvas
def ajusta_curvas(x, y, equacao, only_positive_values, global_parametros_iniciais):
    # Extrai os parâmetros da equação
    parametros = sorted(list(set([str(p) for p in sympify(equacao).free_symbols if str(p) != 'x'])))
    print(parametros)

    # Define a função a ser ajustada
    funcao = sympify(equacao)
    funcao_lambda = lambdify(['x'] + parametros, funcao, 'numpy')

    # Função para o curve_fit
    def funcao_ajuste(x, *params):
        return funcao_lambda(x, *params)

    # Ajuste de curvas

    print("Param:", global_parametros_iniciais)
    if global_parametros_iniciais is not None:
        parametros_iniciais = global_parametros_iniciais
    else:
        parametros_iniciais = np.ones(len(parametros))  # Valores iniciais

    #parametros_iniciais = [20, 600.0, 35.0]

    limite_inferior = [0] * len(parametros)  # Todos os parâmetros >= 0
    limite_superior = [np.inf] * len(parametros)  # Sem limite superior

    try:
        if only_positive_values == ['True']:
            print('Parametros Positivos')
            print(only_positive_values)
            params_opt, params_cov = curve_fit(funcao_ajuste, x, y, p0=parametros_iniciais, bounds=(limite_inferior, limite_superior))
        else:
            print('Parametros Livres')
            print(only_positive_values)
            params_opt, params_cov = curve_fit(funcao_ajuste, x, y, p0=parametros_iniciais)

        mensagem_de_erro = None
    except RuntimeError as e:
        mensagem_de_erro = e
        params_opt = None
        funcao_lambda = None
        parametros = None
    finally:
        return params_opt, funcao_lambda, parametros, mensagem_de_erro

# 4. Plotar o resultado, apresentar o R^2 e a equação ajustada
def plot_resultado(x, y, params_opt, funcao_lambda, parametros, equacao, Output_Columns, log_x_values, log_y_values):
    # Define o caminho do diretório
    directory_path = 'assets/images'

    # Lista todos os arquivos e diretórios no diretório especificado
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        # Verifica se é um arquivo (e não um diretório)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)  # Deleta o arquivo
        else:
            print(f"Ignorado: {file_path} não é um arquivo.")


    # Plotando Gráficos
    i = 0
    y_pred = funcao_lambda(x, *params_opt)
    for output in Output_Columns:
        output = limpar_nome_arquivo(output)
        figure_file = 'assets/images/' + str("%02d" % (i + 1)) + ' - ' + output + '.png'
        plt.figure(1)
        plt.clf()
        plt.scatter(x, y, label='Exp. Data')
        plt.plot(x, y_pred, label='Fit', color='red')

        if log_x_values == ['True']:
            plt.xscale('log')

        if log_y_values == ['True']:
            plt.yscale('log')

        plt.legend()
        plt.savefig(figure_file)
        plt.close()
        i = i + 1

    # Calcula o R^2 usando scikit-learn
    r2 = r2_score(y, y_pred)

    # Mostra a equação ajustada
    equacao_ajustada = 'y = ' + equacao
    for param, valor in zip(parametros, params_opt):
        equacao_ajustada = equacao_ajustada.replace(param, f'{valor:.4f}')

    return r2, equacao_ajustada

def EasyCurveFit(Dataset, Input_Columns, Output_Columns, equation_input,
                 only_positive_values, log_x_values, log_y_values,
                 global_parametros_iniciais):

    Input_Plus_Output = Input_Columns + Output_Columns
    Filtered_Dataset = Dataset[Input_Plus_Output]

    df = Filtered_Dataset
    df = CleanDataset(df)

    x = df[Input_Columns].values
    x = x.squeeze()
    y = df[Output_Columns].values
    y = y.squeeze()

    equacao = solicita_equacao(equation_input)  # O usuário entra com a equação, como 'y = a*x + b'
    #print(equation_input, only_positive_values, log_x_values, log_y_values)
    params_opt, funcao_lambda, parametros, mensagem_de_erro = ajusta_curvas(x, y, equacao, only_positive_values, global_parametros_iniciais)

    if mensagem_de_erro is not None:
        return "", "", mensagem_de_erro
    else:
        r2, equacao_ajustada = plot_resultado(x, y, params_opt, funcao_lambda, parametros, equacao,
                                              Output_Columns, log_x_values, log_y_values)
        return r2, equacao_ajustada, None
