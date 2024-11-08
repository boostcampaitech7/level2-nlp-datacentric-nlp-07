import os
import pandas as pd
from pandas import DataFrame
from typing import List

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')

target_mapping = {
    '생활문화': 0,
    '스포츠': 1,
    '정치': 2,
    '사회': 3,
    'IT과학': 4,
    '기타': 4,
    '경제': 5,
    '세계': 6,
    }

def map_target_number(df: DataFrame) -> DataFrame:
    df['target'] = df['target'].map(target_mapping).fillna(df['target'])
    df['target'] = df['target'].astype(int)
    return df

def print_df_target(df: DataFrame):
    print(f'length : {len(df)}')
    print(df['target'].value_counts())
    print()
    
def read_as_number_target_csv(filename: str) -> DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    df = df.filter(['ID', 'text', 'target'])
    df = map_target_number(df)
    print_df_target(df)

    return df

def concat(dfs: List[DataFrame]) -> DataFrame:
    df = pd.concat(dfs).drop_duplicates(subset='text', keep='first')    
    print_df_target(df)
    return df
