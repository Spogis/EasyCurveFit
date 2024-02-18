import numpy as np
import pandas as pd
import math
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.spatial import distance

from EasyCurveFit.CleanData import *

def SplitCurve(Dataset, Input_Columns, Output_Columns, n_pontos):
    Input_Plus_Output = Input_Columns + Output_Columns
    Filtered_Dataset = Dataset[Input_Plus_Output]

    df = Filtered_Dataset
    df = CleanDataset(df)

    x = df[Input_Columns].values
    x = x.squeeze()
    y = df[Output_Columns].values
    y = y.squeeze()

    original_points = np.vstack((x, y)).T
    df_original = pd.DataFrame(original_points, columns=['X', 'Y'])

    # Criar função de interpolação
    f = interp1d(x, y, kind='cubic')

    # Gerando pontos interpolados (aqui usamos mais pontos para uma curva mais suave)
    x_interpolado = np.linspace(np.min(x), np.max(x), num=n_pontos*1000)
    y_interpolado = f(x_interpolado)
    df_interpolado = pd.DataFrame({'X': x_interpolado, 'Y': y_interpolado })

    def calcular_distancia_euclidiana(x1, y1, x2, y2):
        x1 = float(x1)
        x2 = float(x2)
        y1 = float(y1)
        y2 = float(y2)

        delta_x = abs(x.max() - x.min())
        delta_y = abs(y.max() - y.min())
        theta = math.atan2(delta_y, delta_x)

        p = [x1, y1]
        q = [x2, y2]

        dist = distance.minkowski(p, q, p=2, w=[((delta_y/delta_x))**2, ((delta_x/delta_y))**2])
        #dist = distance.minkowski(p, q, p=2, w=[math.sin(theta)**2, math.cos(theta)**2])

        return dist

    def calcular_distancia_euclidiana_aaa(x1, y1, x2, y2):
        delta_x = abs(x.max() - x.min())
        delta_y = abs(y.max() - y.min())
        theta = math.atan2(delta_y, delta_x)

        dx = x2 - x1
        dy = y2 - y1

        dist = math.sqrt((dx-math.sin(theta))**2 + (dy-math.cos(theta))**2)

        return dist

    # Calculando a soma das distâncias euclidianas entre pontos interpolados consecutivos
    comprimento_total  = sum(calcular_distancia_euclidiana(x1, y1, x2, y2) for x1, y1, x2, y2 in
                          zip(x_interpolado[:-1], y_interpolado[:-1], x_interpolado[1:], y_interpolado[1:]))

    print(f"A soma das distâncias entre os pontos interpolados é: {comprimento_total}")

    # Função para encontrar x que corresponde a um comprimento dado ao longo da curva
    def find_x_for_length(target_length, start_x, end_x):
        x_inicial = float(start_x)
        def objective(a):
            return (calcular_distancia_euclidiana(x_inicial, f(x_inicial), a, f(a)) - target_length)**2

        x0=x_inicial
        limites = [(x_inicial, end_x)]
        #print(limites)
        res = minimize(objective, x0, bounds=limites, method='Powell')
        #print(start_x, res.x)
        return res.x

    # Gera pontos equidistantes ao longo da curva
    segment_length = comprimento_total / n_pontos
    print('segment_length', segment_length)
    x_points = [x.min()]
    for i in range(1, n_pontos):
        next_x = find_x_for_length(segment_length, x_points[i-1], x.max())
        next_x = float(next_x)
        x_points.append(next_x)
    x_points.append(x.max())

    # Calcula y para os novos pontos x
    y_points = f(x_points)

    # Converte os resultados para um DataFrame e salva em um arquivo Excel
    df_novo = pd.DataFrame({'X': x_points, 'Y': y_points})
    df_novo.to_excel('assets/pontos_equidistantes.xlsx', index=False)

    return df_novo, df_original