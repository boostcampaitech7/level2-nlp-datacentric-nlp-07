import pandas as pd
import re

class NoiseDataSeparator:
    """
    NoiseDataSeparator 클래스는 데이터프레임에서 텍스트 열을 처리하여 특수 문자를 제거하고,
    특수 문자의 비율을 계산하여 특정 비율 이상의 데이터를 분리하는 기능을 제공합니다.
    이 클래스는 텍스트 데이터 전처리에서 유용하며, 텍스트에서 잡음(noise)을 제거하고,
    노이즈가 많거나 적은 데이터를 구분하여 저장할 수 있도록 돕습니다.
    """

    def __init__(self, dataframe: pd.DataFrame, text_column: str):
        """
        NoiseDataSeparator 클래스의 생성자. 데이터프레임과 텍스트 열을 받아 초기화합니다.
        
        :param dataframe: 전처리할 데이터프레임
        :param text_column: 텍스트가 포함된 열의 이름
        """
        self.df = dataframe  # 데이터프레임 저장
        self.text_column = text_column  # 텍스트 열 이름 저장

    def replace_symbols(self):
        """
        텍스트 데이터에서 정의된 특수 문자들을 교체하는 메서드.
        
        주요 교체 항목:
        - '…', '...' -> 공백
        - '·' -> 공백
        - '→' -> '에서'
        - '↑' -> '상승'
        - '↓' -> '하락'
        - '↔' -> 공백
        
        :return: self (메서드 체이닝을 위해 반환)
        """
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
        """
        텍스트에서 특수 문자를 찾아 개수와 비율을 계산하여 데이터프레임에 추가하는 메서드.
        
        :param pattern: 특수 문자를 찾을 정규 표현식 패턴 (기본값으로 한글, 알파벳, 숫자 외의 문자)
        :return: self (메서드 체이닝을 위해 반환)
        """
        # 'special_char_count' 열에 특수 문자의 개수 추가
        self.df['special_char_count'] = self.df[self.text_column].apply(lambda text: len(re.findall(pattern, text)))
        # 'special_char_ratio' 열에 특수 문자 비율 추가
        self.df['special_char_ratio'] = self.df['special_char_count'] / self.df[self.text_column].str.len()
        return self

    def filter_by_special_char_ratio(self, threshold: float):
        """
        특수 문자 비율을 기준으로 데이터를 필터링하여 비율이 높은 데이터와 그 외 데이터를 분리합니다.
        
        :param threshold: 특수 문자 비율을 기준으로 분리할 임계값
        :return: 높은 비율의 데이터와 그 외의 데이터를 반환
        """
        # 특수 문자 비율을 기준으로 데이터를 정렬
        df_sorted = self.df.sort_values(by='special_char_ratio', ascending=False)
        # 높은 비율을 가진 데이터
        self.df_high_ratio = df_sorted[df_sorted['special_char_ratio'] >= threshold]
        # 나머지 데이터
        self.df_excluded = self.df[~self.df.index.isin(self.df_high_ratio.index)]
        return self.df_high_ratio, self.df_excluded

    def save_to_csv(self, high_ratio_path: str, noise_free_path: str):
        """
        필터링된 데이터프레임을 CSV 파일로 저장하는 메서드.
        
        :param high_ratio_path: 특수 문자 비율이 높은 데이터프레임을 저장할 파일 경로
        :param noise_free_path: 잡음이 적은 데이터프레임을 저장할 파일 경로
        """
        # 높은 비율 데이터를 high_ratio_path로 저장
        self.df_high_ratio.to_csv(high_ratio_path, index=False, encoding='utf-8-sig')
        # 잡음이 적은 데이터를 noise_free_path로 저장
        self.df_excluded.to_csv(noise_free_path, index=False)


# 사용 예시
if __name__ == "__main__":
    # 데이터 로드
    df = pd.read_csv('../data/train.csv')
    
    # NoiseDataSeparator 클래스 인스턴스 생성
    separator = NoiseDataSeparator(dataframe=df, text_column='text')
    
    # 특수 문자 교체 및 특수 문자 비율 계산 후 필터링
    df_high_ratio, df_excluded = (
        separator
        .replace_symbols()  # 특수 문자 교체
        .add_special_char_metrics()  # 특수 문자 비율 계산
        .filter_by_special_char_ratio(threshold=0.042)  # 임계값으로 필터링
    )
    
    # 결과를 CSV 파일로 저장
    separator.save_to_csv('../data/df_high_ratio.csv', '../data/noise_free.csv')
