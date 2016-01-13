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

def apply_lower(field):
    if isinstance(field, basestring):
        return field.lower()
    elif isinstance(field, (list, np.ndarray)):
        return map(apply_lower, field)
    else:
        return field

def to_lower(row, columns):
    # for col in columns:
    #     field = row[col]
    #     print field
    #
    #     apply_lower(field)

    return row.apply(apply_lower)

def lowerCaseDataset(df, filename):
    columns = df.columns

    def wrapper_func(row):
        return to_lower(row, columns)

    new_df = df.apply(wrapper_func, axis=1)
    writeData(new_df, filename)