import re
import pandas as pd
from .noise_filter import NoiseFilter
from .constants import (
    ENGLISH_COUNT_THRESHOLD, COLUMN_TEXT, COLUMN_ENGLISH_COUNT,
    CONTINUOUS_ENGLISH_LENGTH, COLUMN_CONTINUOUS_ENGLISH
)


class EnglishFilter(NoiseFilter):
    # TODO: 채은한테 리뷰 꼭 받기!
    def filter_noise(self, data: pd.DataFrame) -> pd.DataFrame:
        data[COLUMN_ENGLISH_COUNT] = data[COLUMN_TEXT].str.findall(r'[A-Za-z]+').str.len()
        return data[data[COLUMN_ENGLISH_COUNT] >= ENGLISH_COUNT_THRESHOLD]

class ContinuousEnglishFilter(NoiseFilter):
    def filter_noise(self, data: pd.DataFrame) -> pd.DataFrame:
        def is_continuous_english(text):
            abnormal_pattern = re.compile(rf'[a-zA-Z]{{{CONTINUOUS_ENGLISH_LENGTH},}}')
            matches = abnormal_pattern.findall(text)
            return bool(matches)
        
        data[COLUMN_CONTINUOUS_ENGLISH] = data[COLUMN_TEXT].apply(is_continuous_english)
        return data[data[COLUMN_CONTINUOUS_ENGLISH]]