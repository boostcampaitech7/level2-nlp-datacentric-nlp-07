import pandas as pd
from src.config.path_config import DATA_PATH, ASCII_PATH_SCS, ASCII_PATH_S
from src.config.data_config import BASIC_COLUMNS,ARROW_REPLACEMENTS, SPECIAL_CHAR_PATTERN
import re
from typing import Tuple, List


class DataClassification:
    def __init__(self):
        """
        Initialize DataClassification with file paths and configuration
        
        Args:
            data_path: Path to input data CSV file
            ascii_path_scs: Path to save SCS filtered data
            ascii_path_s: Path to save S filtered data
            basic_columns: List of basic columns to keep in output
        """
        self.data_path = DATA_PATH
        self.ascii_path_scs = ASCII_PATH_SCS
        self.ascii_path_s = ASCII_PATH_S
        self.basic_columns = BASIC_COLUMNS
        self.special_char_pattern = SPECIAL_CHAR_PATTERN
        self.arrow_replacements = ARROW_REPLACEMENTS
        self.data = None
        
    def calculate_ascii(self, text: str) -> Tuple[int, bool, bool, float, float]:
        """
        Calculate ASCII-related metrics for given text
        
        Args:
            text: Input text string
            
        Returns:
            Tuple containing ascii_count, is_english_only, is_all_uppercase, 
            ascii_ratio, uppercase_ratio
        """
        # Get non-space ASCII characters
        ascii_chars = [char for char in text if ord(char) < 128 and not char.isspace()]
        ascii_count = len(ascii_chars)
        ascii_ratio = ascii_count / len(text)

        # English-related checks
        english_chars = [char for char in ascii_chars if char.isalpha()]
        has_english = len(english_chars) > 0
        
        # English-only check if English characters exist
        is_english_only = all(char.isalpha() for char in ascii_chars) if has_english else False
        
        # Uppercase calculations (only if English exists)
        is_all_uppercase = all(char.isupper() for char in english_chars) if has_english else False
        uppercase_count = sum(1 for char in ascii_chars if char.isupper())
        uppercase_ratio = uppercase_count / len(ascii_chars) if ascii_chars else 0
        
        return ascii_count, is_english_only, is_all_uppercase, ascii_ratio, uppercase_ratio
    
    def load_and_preprocess_data(self) -> None:
        """Load data and perform initial text preprocessing"""
        self.data = pd.read_csv(self.data_path)
        
        # Replace specific characters
        replace_chars = ['…', '...', '·']
        for char in replace_chars:
            self.data['text'] = self.data['text'].str.replace(char, ' ', regex=False)
            
    def count_special_characters(self, text: str) -> int:
        """
        텍스트에서 특수 문자를 찾아 개수와 비율을 계산하여 데이터프레임에 추가하는 메서드.
        특수 문자를 찾을 정규 표현식 패턴 (기본값으로 한글, 알파벳, 숫자 외의 문자)
        """
        return len(re.findall(self.special_char_pattern, text))
    
    def replace_arrow_symbols(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace arrow symbols with their corresponding meaning in Korean
        Args:
            df: Input DataFrame containing 'text' column
        Returns:
            pd.DataFrame: DataFrame with arrow symbols replaced in 'text' column
        Notes:
            Replacements:
            - '→' -> '에서'
            - '↑' -> '상승'
            - '↓' -> '하락'
            - '↔' -> 공백
        """
        df = df.copy() 
        for arrow, replacement in self.arrow_replacements.items():
            df['text'] = df['text'].str.replace(arrow, replacement, regex=False)
        return df
    
    def process_data(self) -> None:
        """Process data with ASCII and special character calculations"""
        # Calculate ASCII metrics
        (
            self.data['ascii_count'],
            self.data['is_english_ascii'],
            self.data['is_all_uppercase'],
            self.data['ascii_ratio'],
            self.data['uppercase_ratio']
        ) = zip(*self.data['text'].apply(self.calculate_ascii))

        # Calculate special character metrics
        self.data['special_char_count'] = self.data['text'].apply(self.count_special_characters)
        self.data['special_char_ratio'] = self.data['special_char_count'] / self.data['text'].str.len()
        self.data['english_count'] = self.data['text'].str.findall(r'[A-Za-z]+').str.len()
    
    def filter_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply filtering conditions to the data
        
        Returns:
            tuple: (data_scs, data_s) - Two DataFrames with filtered data
        """
        # Define conditions
        cond1 = (
            (self.data['ascii_ratio'] >= 0.15) & 
            (self.data['special_char_ratio'] >= 0.030) & 
            (self.data['english_count'] >= 2)
        )
        cond2 = self.data['special_char_ratio'] >= 0.042

        # Filter data
        data_scs = self.data[cond1]
        data_s = self.data[cond2]

        # Select basic columns
        data_scs = data_scs[self.basic_columns]
        data_s = data_s[self.basic_columns]
        data_S = self.replace_arrow_symbols(data_s)
        return data_scs, data_s
    
    def save_data(self, data_scs: pd.DataFrame, data_s: pd.DataFrame) -> None:
        """
        Save filtered data to specified paths
        
        Args:
            data_scs: DataFrame containing SCS filtered data
            data_s: DataFrame containing S filtered data
        """
        data_s.to_csv(self.ascii_path_s, index=False)
        data_scs.to_csv(self.ascii_path_scs, index=False)

    def run(self) -> None:
        """Execute the complete data classification process"""
        self.load_and_preprocess_data()
        self.process_data()
        data_scs, data_s = self.filter_data()
        self.save_data(data_scs, data_s)