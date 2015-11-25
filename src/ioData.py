__author__ = 'andrei'

import numpy as np
import pandas as pd

pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', 400)

def readData(filename):
    df = pd.read_json(filename)
    assert isinstance(df, pd.DataFrame)
    df = df.reset_index()
    df = df.drop('index', axis=1)
    return df

def writeData(df, filename):
    assert isinstance(df, pd.DataFrame)
    df.to_json(filename)
    return
