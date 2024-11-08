import pandas as pd


df = pd.read_csv('generate_test.csv')
df = df.drop_duplicates(subset=['generate_title'])
df = df.reset_index(drop=True)
df['ID'] = df.index
df = df.rename(columns={'generate_title': 'text'})
df = df[['ID', 'target', 'text']]
df.to_csv('processed_generate_mark2.csv', index=False)