import pandas as pd
import math
from scipy.interpolate import interp1d
import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objs as go

from EasyCurveFit.CleanData import *

def CreateInterpolatedDataset(Dataset, Input_Columns, Output_Columns):
    global df_interpolado
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
    x_interpolado = np.linspace(np.min(x), np.max(x), num=1000)
    y_interpolado = f(x_interpolado)
    df_interpolado = pd.DataFrame({'x': x_interpolado, 'y': y_interpolado})
    return df_interpolado

def CreateClickLayout(Dataset, Input_Columns, Output_Columns):
    df_interpolado = CreateInterpolatedDataset(Dataset, Input_Columns, Output_Columns)

    click_layout = html.Div([
        dcc.Graph(id='main-graph', figure={
            'data': [go.Scatter(x=df_interpolado['x'], y=df_interpolado['y'], mode='lines', name='Experimental Data')],
            'layout': go.Layout(clickmode='event+select')
        }),
        html.Div([
            html.Button("Mode: Add", id="btn-toggle", n_clicks=0, style={'backgroundColor': 'green', 'color': 'white', 'fontWeight': 'bold', 'fontSize': '20px', 'marginRight': '10px'}),
            html.Button("Download Excel", id="btn-download", n_clicks=0, style={'backgroundColor': 'blue', 'color': 'white', 'fontWeight': 'bold', 'fontSize': '20px'})
        ], style={'padding': '20px'}),
        dcc.Download(id="download-excel")
    ], style={'width': '80%', 'justifyContent': 'center', 'margin-left': 'auto', 'margin-right': 'auto', 'padding': '20px'})

    return click_layout


