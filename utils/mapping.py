target_mapping = {
    '생활문화': 0,
    '스포츠': 1,
    '정치': 2,
    '사회': 3,
    'IT과학': 4,
    '경제': 5,
    '세계': 6
    }

def map_target_number(df):
    df['target'] = df['target'].replace(target_mapping)   
    return df
