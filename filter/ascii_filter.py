import pandas as pd
from .noise_filter import NoiseFilter
from .constants import ASCII_RATIO_THRESHOLD, COLUMN_TEXT, COLUMN_ASCII_RATIO

class AsciiFilter(NoiseFilter):
    def filter_noise(self, data: pd.DataFrame) -> pd.DataFrame:
        def calculate_ascii(text):
            ascii_chars = [char for char in text if ord(char) < 128 and not char.isspace()]
            ascii_ratio = len(ascii_chars) / len(text) if len(text) > 0 else 0
            return ascii_ratio
        
        data[COLUMN_ASCII_RATIO] = data[COLUMN_TEXT].apply(calculate_ascii)
        return data[data[COLUMN_ASCII_RATIO] >= ASCII_RATIO_THRESHOLD]
