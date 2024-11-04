import re
import pandas as pd
from .noise_filter import NoiseFilter
from .constants import (
    SPECIAL_CHAR_RATIO_THRESHOLD, SPECIAL_CHAR_PATTERN,
    COLUMN_TEXT, COLUMN_SPECIAL_CHAR_COUNT, COLUMN_SPECIAL_CHAR_RATIO
)


class SpecialCharFilter(NoiseFilter):
    def filter_noise(self, data: pd.DataFrame) -> pd.DataFrame:
        def count_special_characters(text):
            return len(re.findall(SPECIAL_CHAR_PATTERN, text))
        
        data[COLUMN_SPECIAL_CHAR_COUNT] = data[COLUMN_TEXT].apply(count_special_characters)
        data[COLUMN_SPECIAL_CHAR_RATIO] = data[COLUMN_SPECIAL_CHAR_COUNT] / data[COLUMN_TEXT].str.len()
        return data[data[COLUMN_SPECIAL_CHAR_RATIO] >= SPECIAL_CHAR_RATIO_THRESHOLD]
