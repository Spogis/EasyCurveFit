# Importações necessárias
import base64
import io
import time
from io import BytesIO

import dash
from dash.dependencies import Input, Output, State
import utils.dash_reusable_components as drc
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash import dash_table
import plotly.graph_objs as go

from EasyCurveFit.CurveFit import *
from EasyCurveFit.CurvePrep import *
from EasyCurveFit.Brent import *
from EasyCurveFit.ClickPoints import *

Input_Columns = None
Output_Columns = None
Dataset = None
global_parametros_iniciais = None

# Lista para armazenar os pontos clicados
clicked_points = []

# Inicializa o app Dash
app = dash.Dash(__name__,
                suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}],)

app.title = "Easy Curve Fit"
server = app.server

app.layout = html.Div([
    html.Br(),
    html.Br(),
    html.Div([
        html.Img(src='assets/logo.png', style={'height': '100px', 'margin-left': 'auto', 'margin-right': 'auto'}),
    ], style={'text-align': 'center', 'margin-bottom': '10px'}),

    html.Div([
        dcc.Tabs(id='tabs', value='tab1', children=[
            dcc.Tab(label='Data', value='tab1'),
            dcc.Tab(label='Curve Fit', value='tab2'),
            dcc.Tab(label='Curve Prep', value='tab3'),
            dcc.Tab(label='Extract Points', value='tab4'),
            dcc.Tab(label='About', value='tab5'),
        ], style={'align': 'center', 'width': '80%', 'margin-left': 'auto', 'margin-right': 'auto'}),
    ]),
    dcc.Store(id='store', storage_type='memory'),
    html.Div(id='tabs-content'),
])

dataset_layout = html.Div([
    html.Br(),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select an Excel or CSV File (Your Dataset)')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
        },
        multiple=False
    ),
    html.Br(),
    html.Label('Select which will be the column of independent data (X):'),
    dcc.Dropdown(
        id='column-input-selector',
        multi=True,
        placeholder='Select the columns after loading a file'
    ),
    html.Br(),
    dash_table.DataTable(
        id='input-table',
        page_size=3,
    ),
    html.Br(),
    html.Label('Select which will be the column of dependent data (Y):'),
    dcc.Dropdown(
        id='column-output-selector',
        multi=True,
        placeholder='Select the columns after loading a file'
    ),
    html.Br(),
    dash_table.DataTable(
        id='output-table',
        page_size=3,
    ),
], style={'width': '80%', 'justifyContent': 'center', 'margin-left': 'auto', 'margin-right': 'auto', 'padding': '20px'})

simple_layout = html.Div([

    html.H5('Load Predefined Model or Custom Model?'),
    dcc.Dropdown(
        id='fit-model',
        options=[
            {'label': 'Custom Model', 'value': 'Custom Model'},
            {'label': 'Linear Model', 'value': 'Linear Model'},
            {'label': 'Exponential Model', 'value': 'Exponential Model'},
            {'label': 'First Order Model', 'value': 'First Order Model'},
            {'label': 'Generalized Logistic Function - Richards Curve', 'value': 'Generalized Logistic Function - Richards Curve'},
            {'label': 'Granulometric Distribution', 'value': 'Granulometric Distribution'},
            {'label': 'Nagata', 'value': 'Nagata'},
            {'label': 'Peak', 'value': 'Peak'},
           ],
        value='Custom Model',
        multi=False,
    ),
    html.Br(),
    html.Div([
        html.Div([
            dcc.Checklist(
                id='only-positive-values',
                options=[
                    {'label': 'Only Positive Parameters', 'value': 'True', 'fontSize': '20px'}
                ],
                value=['True'],
            )
        ], style={'margin-right': '20px'}),  # Adiciona margem à direita para este div

        html.Div([
            dcc.Checklist(
                id='log-x-values',
                options=[
                    {'label': 'X-axis - Logarithmic Scale', 'value': 'True', 'fontSize': '20px'}
                ],
                value=[],
            )
        ], style={'margin-right': '20px'}),  # Adiciona margem à direita para este div

        html.Div([
            dcc.Checklist(
                id='log-y-values',
                options=[
                    {'label': 'Y-axis - Logarithmic Scale', 'value': 'True', 'fontSize': '20px'}
                ],
                value=[],
            )
        ]),  # Não adiciona margem ao último elemento
    ], style={'display': 'flex', 'width': '100%', 'justifyContent': 'center', 'alignItems': 'center',
              'margin-left': '10px', 'margin-right': '10px', 'padding': '20px'}),

    html.Div([
        html.Div([
            html.H5("Curve Fit Model:"),
            dcc.Textarea(
                id='equation_input',
                style={'width': '100%', 'height': 50, 'resize': 'none', 'color': 'white', 'fontWeight': 'bold', 'fontSize': '20px'},
                readOnly=False
            ),
        ]),  # Não adiciona margem ao último elemento

        html.Div([
            html.H5("Initial Parameters Values:"),
            dcc.Textarea(
                id='initial_parameter_values',
                style={'width': '100%', 'height': 50, 'resize': 'none', 'color': 'white', 'fontWeight': 'bold', 'fontSize': '20px'},
                readOnly=False
            ),
        ], id='div-initial_parameter-oculto', style={'display': 'none'}),
    ]),

    html.Div([
        html.Div([
            html.Button('INITIAL PARAMETERS!',
                                id='initial-parameters-button',
                                disabled=False,
                                style={'display': 'flex', 'width': '300px', 'justifyContent': 'center',
                                       'color': 'white', 'fontWeight': 'bold', 'background-color': 'green',
                                       'margin-left': 'auto', 'margin-right': 'auto',
                                       'margin-top': '10px', 'margin-bottom': '10px'}),
        ], style={'margin-right': '20px'}),  # Adiciona margem à direita para este div

        html.Div([
            html.Button('RUN CURVE FIT!',
                                id='run-MLP-button',
                                disabled=False,
                                style={'display': 'flex', 'width': '300px', 'justifyContent': 'center',
                                       'color': 'white', 'fontWeight': 'bold', 'background-color': 'green',
                                       'margin-left': 'auto', 'margin-right': 'auto',
                                       'margin-top': '10px', 'margin-bottom': '10px'}),
        ]),  # Não adiciona margem ao último elemento
    ], style={'display': 'flex', 'width': '100%', 'justifyContent': 'center', 'alignItems': 'center',
                  'margin-left': '10px', 'margin-right': '10px', 'padding': '20px'}),

    dbc.Spinner(html.Div(id="loading-output1"), spinner_style={"width": "3rem", "height": "3rem"}),
    html.H5("Curve Fit Results:"),
    dcc.Textarea(
        id='r2-simple-mlp-textarea',
        style={'width': '100%', 'height': 100, 'resize': 'none', 'color': 'white', 'fontWeight': 'bold', 'fontSize': '20px'},
        readOnly=True
    ),
    html.Br(),
    html.Div([
        dash_table.DataTable(id='table-adjust',
                             columns=[{"name": "Parameter", "id": "Parameter"},
                                      {"name": "Adjusted Value", "id": "Adjusted Value"},
                                      {"name": "Standard Deviation", "id": "Standard Deviation"}],
                             data=[],
                             style_cell={'textAlign': 'left', 'fontSize': '20px', 'fontFamily': 'Arial'},
        )
    ]),
    html.Br(),
    html.Div([
        html.Div(id='button-output'),
    ], style={'width': '100%', 'textAlign': 'center', 'justifyContent': 'center', 'margin-left': 'auto', 'margin-right': 'auto'}),

], style={'width': '80%', 'justifyContent': 'center', 'margin-left': 'auto', 'margin-right': 'auto', 'padding': '20px'})

# Carregar o conteúdo do arquivo README.md
with open('README.md', 'r', encoding='utf-8') as file:  # Garantindo a leitura correta de caracteres
    readme_content = file.read()

about_layout = html.Div([
    dcc.Markdown(children=readme_content)
], style={'width': '80%', 'justifyContent': 'center', 'margin-left': 'auto', 'margin-right': 'auto', 'padding': '20px'})

# Curve Prep Layout
curve_prep_layout = html.Div([
    html.Div([
        html.Div([
            drc.NamedSlider(
                name="Number of Points",
                id="ramer_douglas_peucker_epsilon",
                min=1,
                max=3,
                step=1,
                marks={
                    1: 'Coarse',
                    2: 'Balanced',
                    3: 'Fine',
                },
                included=False,
                value=2
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'margin': '20px', 'textAlign': 'center'}),

        html.Div([
            html.Button('Download', id='botao-download', n_clicks=0,
                        style={'margin': '20px', 'color': 'white', 'fontWeight': 'bold', 'display': 'inline-block'}),
        ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'center'}),

    ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}),

    dcc.Graph(id='meu-grafico'),

    html.H5("Filtered Curve Results:"),
    dcc.Textarea(
        id='filtered-curve-textarea',
        style={'width': '100%', 'height': 200, 'resize': 'none', 'color': 'white', 'fontWeight': 'bold'},
        readOnly=True
    ),

    dcc.Download(id="download-curve"),

], style={'width': '80%', 'justifyContent': 'center', 'margin-left': 'auto', 'margin-right': 'auto', 'padding': '20px'})



def parse_contents(contents, filename):
    global Dataset
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'xlsx' in filename:
            # Assume que é um arquivo Excel
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'csv' in filename:
            # Assume que é um arquivo CSV
            df = pd.read_csv(io.BytesIO(decoded))
        else:
            return html.Div([
                'Unsupported file type.'
            ])
    except Exception as e:
        return html.Div([
            'There was an error processing the file.'
        ])

    Dataset = df

    return df

@app.callback(
    Output("download-curve", "data"),
    Input("botao-download", "n_clicks"),
    prevent_initial_call=True
)

def download(n_clicks):
    file_path = 'assets/Filtered_RDP.xlsx'
    return dcc.send_file(file_path)


# Callback para atualizar o gráfico
@app.callback(
    [Output('meu-grafico', 'figure'),
     Output('filtered-curve-textarea', 'value')],
    Input('ramer_douglas_peucker_epsilon', 'value')
)

def atualizar_grafico(valor_slider):
    if valor_slider == 1:
        epsilon = 0.01
        n_pontos = 10
    if valor_slider == 2:
        epsilon = 0.005
        n_pontos = 30
    if valor_slider == 3:
        epsilon = 0.001
        n_pontos = 50

    RDP_Return_String, df_original, df_simplified, df_gaussian = RDP(Dataset, Input_Columns, Output_Columns, epsilon)
    #df_novo, df_original = SplitCurve(Dataset, Input_Columns, Output_Columns, n_pontos)

    # Criação do gráfico com os dois datasets
    fig = go.Figure()

    # Dados originais
    fig.add_trace(go.Scatter(x=df_original['X'], y=df_original['Y'], mode='lines', name='Experimental Data'))

    # Dados filtro Gaussiano
    # fig.add_trace(go.Scatter(x=df_gaussian['X'], y=df_gaussian['Y'], mode='lines', name='Gaussian Filter'))

    # Dados Brent
    #fig.add_trace(go.Scatter(x=df_novo['X'], y=df_novo['Y'], mode='markers', name='Brent Points'))

    # Dados filtrados
    fig.add_trace(go.Scatter(x=df_simplified['X'], y=df_simplified['Y'], mode='markers', name='Filtered Data'))

    # Atualiza layout do gráfico
    fig.update_layout(xaxis_title='X',
                      yaxis_title='Y',
                      legend=dict(orientation="h",
                                  x=0.5,
                                  y=1.1,
                                  xanchor="center",
                                  yanchor="bottom")
                      )
    return fig, RDP_Return_String

@app.callback([Output('initial_parameter_values', 'value', allow_duplicate=True),
               Output('div-initial_parameter-oculto', 'style')],
              Input('initial_parameter_values', 'value'),
              prevent_initial_call=True)

def EstimateInitialParameters(initial_parameter_values):
    global global_parametros_iniciais
    def extrair_valores_da_string(entrada):
        elementos = entrada.split(', ')
        valores = [float(elemento.split('=')[1]) for elemento in elementos]
        return valores

    # Usando a função para obter os valores numéricos da string
    global_parametros_iniciais = extrair_valores_da_string(initial_parameter_values)
    return initial_parameter_values, {'display': 'block'}

@app.callback(Output('initial_parameter_values', 'value', allow_duplicate=True),
              State('equation_input', 'value'),
              Input('initial-parameters-button', 'n_clicks'),
              prevent_initial_call=True)

def EstimateInitialParameters(equation_input, n_clicks):
    parametros = ParametersData(equation_input)

    def format_array(values):
        formatted_values = [f"{value}=1.0 " for value in values]
        return ', '.join(formatted_values)

    resultado = format_array(parametros)

    return resultado

@app.callback(Output('equation_input', 'value', allow_duplicate=True),
              Input('fit-model', 'value'),
              prevent_initial_call=True)

def WriteEquation(fit_model):
    if fit_model == 'Custom Model':
        return ""
    elif fit_model == 'Linear Model':
        return 'y=a*x+b'
    elif fit_model == 'Exponential Model':
        return 'y=a*x*exp(b*x)'
    elif fit_model == 'First Order Model':
        return 'y=𝐾*(1-exp(-(1/𝜏)*x))'
    elif fit_model == 'Generalized Logistic Function - Richards Curve':
        return 'y=a+((k-a)/(c+q*exp(-b*x))**(1/v))'
    elif fit_model == 'Granulometric Distribution':
        return 'y=1-exp(-(x/D)**n)'
    elif fit_model == 'Nagata':
        return 'y=(a/x) + b*((10**3 + 0.6*f*(x**c)) / (10**3 + 1.6*f*(x**c)))**p'
    elif fit_model == 'Peak':
        return 'y=(b/(sqrt(1+a*((k-x)**2))))'

@app.callback(Output('equation_input', 'value'),
              Input('equation_input', 'value'))

def update_equation(equation):
    return equation

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])

def update_tab_content(selected_tab):
    if selected_tab == 'tab1':
        return dataset_layout
    elif selected_tab == 'tab2':
        return simple_layout
    elif selected_tab == 'tab3':
        return curve_prep_layout
    elif selected_tab == 'tab4':
        return CreateClickLayout(Dataset, Input_Columns, Output_Columns)
    elif selected_tab == 'tab5':
        return about_layout

@app.callback(
    [Output("loading-output1", "children", allow_duplicate=True),
     Output("button-output", "children", allow_duplicate=True),
     Output('r2-simple-mlp-textarea', 'value'),
     Output('table-adjust', 'data')],
    State('equation_input', 'value'),
    State('fit-model', 'value'),
    State('only-positive-values', 'value'),
    State('log-x-values', 'value'),
    State('log-y-values', 'value'),
    Input("run-MLP-button", "n_clicks"),
    prevent_initial_call=True
)

def CurveFit(equation_input, fit_model, only_positive_values, log_x_values, log_y_values, n_clicks):
    global global_parametros_iniciais

    if fit_model == 'Custom Model':
        equation_input = equation_input
    elif fit_model == 'Linear Model':
        equation_input = 'y=a*x+b'
    elif fit_model == 'Exponential Model':
        equation_input = 'y=a*x*exp(b*x)'
    elif fit_model == 'First Order Model':
        equation_input = 'y=𝐾*(1-exp(-(1/𝜏)*x))'
    elif fit_model == 'Generalized Logistic Function - Richards Curve':
        equation_input = 'y=a+((k-a)/(c+q*exp(-b*x))**(1/v))'
    elif fit_model == 'Granulometric Distribution':
        equation_input = 'y=1-exp(-(x/D)**n)'
    elif fit_model == 'Nagata':
        equation_input = 'y=(a/x) + b*((10**3 + 0.6*f*(x**c)) / (10**3 + 1.6*f*(x**c)))**p'
    elif fit_model == 'Peak':
        equation_input = 'y=(b/(sqrt(1+a*((k-x)**2))))'

    r2_str, equacao_ajustada_str, mensagem_de_erro, parametros, params_opt, desvios = EasyCurveFit(Dataset, Input_Columns, Output_Columns, equation_input,
                                                                                                   only_positive_values, log_x_values, log_y_values,
                                                                                                   global_parametros_iniciais)

    directory_path = 'assets/images'
    image_components = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']

    for filename in os.listdir(directory_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            file_path = os.path.join(directory_path, filename)
            unique_path = f"{file_path}?t={int(time.time())}"
            image_components.append(html.Img(src=unique_path, style={'width': '50%', 'height': 'auto'}))

    loading_status = ""

    if mensagem_de_erro is not None:
        texto_Retorno = str(mensagem_de_erro)
    else:
        texto_Retorno = f"{equacao_ajustada_str} \n\n"
        valor_formatado = f"{r2_str:.4f}"
        texto_Retorno += f"r²: {valor_formatado}"

        df_adjust = pd.DataFrame({
            "Parameter": parametros,
            "Adjusted Value": params_opt,
            "Standard Deviation": desvios
        })

    return loading_status, image_components, texto_Retorno, df_adjust.to_dict('records')

@app.callback(
    Output('column-input-selector', 'options'),
    Output('column-input-selector', 'value'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_dropdown(list_of_contents, list_of_names):
    if list_of_contents is not None:
        df = parse_contents(list_of_contents, list_of_names)
        return [{'label': col, 'value': col} for col in df.columns], df.columns.tolist()
    return [], []

@app.callback(
    Output('column-output-selector', 'options'),
    Output('column-output-selector', 'value'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_dropdown(list_of_contents, list_of_names):
    if list_of_contents is not None:
        df = parse_contents(list_of_contents, list_of_names)
        return [{'label': col, 'value': col} for col in df.columns], df.columns.tolist()
    return [], []

@app.callback(
    Output('input-table', 'columns'),
    Output('input-table', 'data'),
    Input('column-input-selector', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_table(selected_columns, list_of_contents, list_of_names):
    global Input_Columns
    if list_of_contents is not None and selected_columns is not None:
        df = parse_contents(list_of_contents, list_of_names)
        filtered_df = df[selected_columns]
        columns = [{"name": col, "id": col} for col in filtered_df.columns]
        data = filtered_df.to_dict('records')
        Input_Columns = selected_columns
        return columns, data
    return [], []

@app.callback(
    Output('output-table', 'columns'),
    Output('output-table', 'data'),
    Input('column-output-selector', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_table(selected_columns, list_of_contents, list_of_names):
    global Output_Columns
    if list_of_contents is not None and selected_columns is not None:
        df = parse_contents(list_of_contents, list_of_names)
        filtered_df = df[selected_columns]
        columns = [{"name": col, "id": col} for col in filtered_df.columns]
        data = filtered_df.to_dict('records')
        Output_Columns = selected_columns
        return columns, data
    return [], []



########################################################################################################################
# Callback para alternar o modo de operação e atualizar o estilo do botão
@app.callback(
    [Output('btn-toggle', 'children'), Output('btn-toggle', 'style'), Output('btn-toggle', 'n_clicks')],
    [Input('btn-toggle', 'n_clicks'), Input('btn-clear', 'n_clicks')],
    [State('btn-toggle', 'n_clicks')]
)

def toggle_mode(btn_toggle_clicks, btn_clear_clicks, state_toggle_clicks):
    triggered_by = ctx.triggered_id
    if triggered_by == "btn-clear":
        # Resetar para o modo "add" ao clicar em "Clear Points"
        return "Mode: Add", {'backgroundColor': 'green', 'color': 'white', 'fontWeight': 'bold', 'fontSize': '20px',
                             'marginRight': '10px'}, 0
    else:
        mode = 'del' if btn_toggle_clicks % 2 else 'add'
        btn_text = "Mode: Delete" if mode == 'del' else "Mode: Add"
        btn_style = {'backgroundColor': 'red', 'color': 'white', 'fontWeight': 'bold', 'fontSize': '20px',
                     'marginRight': '10px'} if mode == 'del' else {'backgroundColor': 'green', 'color': 'white',
                                                                   'fontWeight': 'bold', 'fontSize': '20px',
                                                                   'marginRight': '10px'}
        return btn_text, btn_style, state_toggle_clicks

# Callback para gerar e baixar o Excel
@app.callback(
    Output("download-excel", "data"),
    Input("btn-download", "n_clicks"),
    prevent_initial_call=True
)
def generate_excel(n_clicks):
    df_points = pd.DataFrame(clicked_points, columns=['x', 'y'])
    df_points = df_points.sort_values(by='x', ascending=True)
    output = BytesIO()
    df_points.to_excel(output, index=False, sheet_name='Filtered Points')
    output.seek(0)
    return dcc.send_bytes(output.getvalue(), filename="filtered_points.xlsx")


# Callback para adicionar ou remover pontos e atualizar o gráfico
@app.callback(
    Output('main-graph', 'figure'),
    [Input('main-graph', 'clickData'),
     Input('btn-toggle', 'n_clicks'),
     Input('btn-clear', 'n_clicks')],
    [State('main-graph', 'figure')]
)
def update_graph(clickData, btn_toggle_clicks, btn_clear_clicks, figure):
    df_interpolado = CreateInterpolatedDataset(Dataset, Input_Columns, Output_Columns)

    # Verifica o contexto do callback para determinar a entrada que acionou a atualização
    if not ctx.triggered or ctx.triggered[0]['prop_id'] == 'btn-toggle.n_clicks':
        raise dash.exceptions.PreventUpdate

    global clicked_points
    if ctx.triggered[0]['prop_id'] == 'btn-clear.n_clicks':
        clicked_points = []  # Limpa todos os pontos clicados
    else:
        mode = 'del' if btn_toggle_clicks % 2 else 'add'
        if clickData:
            x_val, y_val = clickData['points'][0]['x'], clickData['points'][0]['y']
            if mode == 'add':
                clicked_points.append((x_val, y_val))

            elif mode == 'del':
                # Encontrar e remover o ponto mais próximo
                if clicked_points:
                    closest_point = min(clicked_points,
                                        key=lambda point: (point[0] - x_val) ** 2 + (point[1] - y_val) ** 2)
                    clicked_points.remove(closest_point)

    # Atualiza o gráfico com todos os pontos clicados
    figure['data'] = [go.Scatter(x=df_interpolado['x'], y=df_interpolado['y'], mode='lines', name='Experimental Data')] + \
                     [go.Scatter(x=[p[0] for p in clicked_points], y=[p[1] for p in clicked_points], mode='markers', marker=dict(color='red', size=10), name='Filtered Points')]

    return figure

# Roda o app
if __name__ == '__main__':
    app.run_server(debug=False)

##git rm --cached -r .idea