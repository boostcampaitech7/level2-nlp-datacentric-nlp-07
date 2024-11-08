import pandas as pd
import re

class TextCleaner:
    def __init__(self, dataframe: pd.DataFrame, text_column: str):
        self.df = dataframe
        self.text_column = text_column

    def replace_symbols(self):
        replacements = {
            '…': ' ',
            '...': ' ',
            '·': ' ',
            '→': '에서',
            '↑': '상승',
            '↓': '하락',
            '↔': ' '
        }
        for symbol, replacement in replacements.items():
            self.df[self.text_column] = self.df[self.text_column].str.replace(symbol, replacement, regex=False)
        return self

    def add_special_char_metrics(self, pattern: str = r'(?<!\d)\.(?!\d)|(?<!\d)%|[^가-힣A-Z\u4E00-\u9FFF\s0-9\.%㎜㎡]'):
        self.df['special_char_count'] = self.df[self.text_column].apply(lambda text: len(re.findall(pattern, text)))
        self.df['special_char_ratio'] = self.df['special_char_count'] / self.df[self.text_column].str.len()
        return self

    def filter_by_special_char_ratio(self, threshold: float):
        df_sorted = self.df.sort_values(by='special_char_ratio', ascending=False)
        self.df_high_ratio = df_sorted[df_sorted['special_char_ratio'] >= threshold]
        self.df_excluded = self.df[~self.df.index.isin(self.df_high_ratio.index)]
        return self.df_high_ratio, self.df_excluded

    def save_to_csv(self, high_ratio_path: str, noise_free_path: str):
        self.df_high_ratio.to_csv(high_ratio_path, index=False, encoding='utf-8-sig')
        self.df_excluded.to_csv(noise_free_path, index=False)


# 사용 예시
if __name__ == "__main__":
    df = pd.read_csv('../data/train.csv')
    cleaner = TextCleaner(dataframe=df, text_column='text')
    
    df_high_ratio, df_excluded = (
        cleaner
        .replace_symbols()
        .add_special_char_metrics()
        .filter_by_special_char_ratio(threshold=0.042)
    )
    
    cleaner.save_to_csv('../data/df_high_ratio.csv', '../data/noise_free.csv')
