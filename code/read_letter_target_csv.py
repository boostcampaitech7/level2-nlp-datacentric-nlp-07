import os
import pandas as pd
from mapping import map_target_number

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')

def print_df(df):
    print(f'length : {len(df)}')
    print(df['target'].value_counts())
    print()
    
def read_letter_target_csv(filename):
    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    df = map_target_number(df)
    
    print_df(df)

    return df

def concat(dfs):
    df = pd.concat(dfs).drop_duplicates(subset='text', keep='first')    
    print_df(df)
    return df
