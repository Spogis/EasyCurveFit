import numpy as np
import pandas as pd

def CleanDataset(df):

    # Imprimindo o tipo de dados de cada coluna
    df = df.dropna()
    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    df = df.dropna(axis=0)
    df.to_excel("assets/Dataset_No_Outliers.xlsx")

    return df
