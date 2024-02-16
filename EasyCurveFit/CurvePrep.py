import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import gaussian_filter1d

from EasyCurveFit.CleanData import *

def gaussian_filter(data, sigma):
    return gaussian_filter1d(data, sigma)

def ramer_douglas_peucker(points, epsilon):
    def find_farthest_point(segment_start, segment_end, points):
        line_vector = segment_end - segment_start
        line_length = np.linalg.norm(line_vector)
        line_unit_vector = line_vector / line_length if line_length != 0 else 0
        point_vector = points - segment_start
        line_point_proj_length = np.dot(point_vector, line_unit_vector)
        line_point_proj = np.outer(line_point_proj_length, line_unit_vector)
        normal_vector = point_vector - line_point_proj
        distances = np.linalg.norm(normal_vector, axis=1)
        farthest_point_index = np.argmax(distances)
        return farthest_point_index, distances[farthest_point_index]

    if len(points) < 3:
        return points
    farthest_point_index, distance = find_farthest_point(points[0], points[-1], points[1:-1])
    if distance > epsilon:
        left_simplified = ramer_douglas_peucker(points[:farthest_point_index+2], epsilon)
        right_simplified = ramer_douglas_peucker(points[farthest_point_index+1:], epsilon)
        return np.vstack((left_simplified[:-1], right_simplified))
    else:
        return np.vstack((points[0], points[-1]))


def calculate_curvature(x, y):
    t = np.linspace(0, 1, len(x))
    spl_x = UnivariateSpline(t, x, k=4, s=0)
    spl_y = UnivariateSpline(t, y, k=4, s=0)

    dxdt = spl_x.derivative()
    dydt = spl_y.derivative()
    d2xdt2 = dxdt.derivative()
    d2ydt2 = dydt.derivative()

    curvature = np.abs(d2xdt2(t) * dydt(t) - dxdt(t) * d2ydt2(t)) / np.power(dxdt(t)**2 + dydt(t)**2, 1.5)
    return curvature


def RDP(Dataset, Input_Columns, Output_Columns, epsilon):
    Input_Plus_Output = Input_Columns + Output_Columns
    Filtered_Dataset = Dataset[Input_Plus_Output]

    df = Filtered_Dataset
    df = CleanDataset(df)

    x = df[Input_Columns].values
    x = x.squeeze()
    y = df[Output_Columns].values
    y = y.squeeze()

    #x = df[Input_Columns].to_numpy()
    #y = df[Output_Columns].to_numpy()

    points = np.vstack((x, y)).T

    simplified_points = ramer_douglas_peucker(points, epsilon)

    # Para calcular o mse e o rmse, precisamos interpolar os dados simplificados para compará-los com os dados originais
    x_simplified = simplified_points[:, 0]
    y_simplified = simplified_points[:, 1]
    interpolator = interp1d(x_simplified, y_simplified, kind='linear', fill_value='extrapolate')
    y_interpolated = interpolator(x)  # Interpolação dos valores y para os x originais

    # Calculando MSE
    mse = mean_squared_error(y, y_interpolated)
    rmse = np.sqrt(mse)

    # Calculando a Distância Hausdorff
    hausdorff_distance = max(directed_hausdorff(points, simplified_points)[0], directed_hausdorff(simplified_points, points)[0])

    # Cálculo da curvatura para os dados originais e simplificados
    curvature_original = calculate_curvature(x, y)
    curvature_simplified = calculate_curvature(x_simplified, y_simplified)

    # Comparação das curvaturas
    curvature_difference = np.abs(curvature_original - np.interp(x, x_simplified, curvature_simplified))

    # Convertendo para DataFrame
    df_simplified = pd.DataFrame(simplified_points, columns=['X', 'Y'])
    df_original = pd.DataFrame(points, columns=['X', 'Y'])

    # Salvando no arquivo Excel
    output_filename = 'assets/Filtered_RDP.xlsx'
    df_simplified.to_excel(output_filename, index=False)

    RDP_Return_String  = f"Original number of points: {len(points)}\n"
    RDP_Return_String += f"Reduced number of points: {len(simplified_points)} \n\n"

    RDP_Return_String +=f"MSE between original and simplified data: {mse:.2e}\n"
    RDP_Return_String +=f"RMSE between original and simplified data: {rmse:.2e}\n"
    RDP_Return_String +=f"Hausdorff Distance between original and simplified data: {hausdorff_distance:.2e}\n"
    RDP_Return_String +=f"Average Curvature Difference: {np.mean(curvature_difference):.2e}\n"

    return RDP_Return_String, df_original, df_simplified

