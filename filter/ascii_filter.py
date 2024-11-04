import pandas as pd
from .noise_filter import NoiseFilter
from .constants import (
    ASCII_RATIO_THRESHOLD, COLUMN_TEXT, COLUMN_ASCII_RATIO,
    COLUMN_ASCII_COUNT, COLUMN_IS_ENGLISH_ONLY, COLUMN_IS_ALL_UPPERCASE,
    COLUMN_ASCII_UPPERCASE_RATIO, COLUMN_ENGLISH_UPPERCASE_RATIO, COLUMN_LENGTH
)

class AsciiFilter(NoiseFilter):
    def _clean_text(self, text: str) -> str:
        replacements = {'…': ' ', '...': ' ', '·': ' '}
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    def _calculate_ascii(self, text: str) -> dict:
        # TODO: 승범한테 리뷰 꼭 받기!!
        # 공백을 제외한 ASCII 문자들
        ascii_chars = [char for char in text if ord(char) < 128 and not char.isspace()]
        ascii_count = len(ascii_chars)
        ascii_ratio = ascii_count / len(text)

        # ASCII 문자들 중 영어 문자만 추출
        english_chars = [char for char in ascii_chars if char.isalpha()]
        has_english = len(english_chars) > 0
        
        # 영어 관련 체크
        is_english_only = all(char.isalpha() for char in ascii_chars) if has_english else False
        
        # 대문자 관련 계산
        is_all_uppercase = all(char.isupper() for char in english_chars) if has_english else False
        
        # ASCII 문자 중 대문자 비율
        ascii_uppercase_count = sum(1 for char in ascii_chars if char.isupper())
        ascii_uppercase_ratio = ascii_uppercase_count / len(ascii_chars) if ascii_chars else 0
        
        # 영어 문자 중 대문자 비율
        english_uppercase_count = sum(1 for char in english_chars if char.isupper())
        english_uppercase_ratio = english_uppercase_count / len(english_chars) if english_chars else 0
        return {
            COLUMN_ASCII_COUNT: ascii_count,
            COLUMN_IS_ENGLISH_ONLY: is_english_only,
            COLUMN_IS_ALL_UPPERCASE: is_all_uppercase,
            COLUMN_ASCII_RATIO: ascii_ratio,
            COLUMN_ASCII_UPPERCASE_RATIO: ascii_uppercase_ratio,
            COLUMN_ENGLISH_UPPERCASE_RATIO: english_uppercase_ratio            
        }
        
    def filter_noise(self, data: pd.DataFrame) -> pd.DataFrame:
        data[COLUMN_TEXT] = data[COLUMN_TEXT].apply(self._clean_text)
        data[COLUMN_LENGTH] = data[COLUMN_TEXT].str.len()
        ascii_metrics = data[COLUMN_TEXT].apply(self._calculate_ascii).apply(pd.Series)

        data = pd.concat([data, ascii_metrics], axis=1)
        return data[data[COLUMN_ASCII_RATIO] >= ASCII_RATIO_THRESHOLD]
