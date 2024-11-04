import re
import pandas as pd
from .noise_filter import NoiseFilter
from .constants import (
    SPECIAL_CHAR_RATIO_THRESHOLD, SPECIAL_CHAR_PATTERN, ARROWS,
    COLUMN_TEXT, COLUMN_SPECIAL_CHAR_COUNT, COLUMN_SPECIAL_CHAR_RATIO, PERCENT_PATTERN
)


class SpecialCharFilter(NoiseFilter):    
    def _replace_dots(self, df: pd.DataFrame) -> pd.DataFrame:
        df['text'] = df['text'].str.replace('…', ' ', regex=False)
        df['text'] = df['text'].str.replace('...', ' ', regex=False)
        df['text'] = df['text'].str.replace('·', ' ', regex=False)
        return df
    
    def _replace_arrow(self, df: pd.DataFrame) -> pd.DataFrame:
        df['text'] = df['text'].str.replace('→', '에서', regex=False)
        df['text'] = df['text'].str.replace('↑', '상승', regex=False)
        df['text'] = df['text'].str.replace('↓', '하락', regex=False)
        df['text'] = df['text'].str.replace('↔', ' ', regex=False)
        return df
    
    def filter_noise(self, data: pd.DataFrame) -> pd.DataFrame:
        def count_special_characters(text):
            return len(re.findall(SPECIAL_CHAR_PATTERN, text))
        
        data = self._replace_dots(data)
        data = self._replace_arrow(data)
        
        data[COLUMN_SPECIAL_CHAR_COUNT] = data[COLUMN_TEXT].apply(count_special_characters)
        data[COLUMN_SPECIAL_CHAR_RATIO] = data[COLUMN_SPECIAL_CHAR_COUNT] / data[COLUMN_TEXT].str.len()
        
        return data[data[COLUMN_SPECIAL_CHAR_RATIO] >= SPECIAL_CHAR_RATIO_THRESHOLD]

