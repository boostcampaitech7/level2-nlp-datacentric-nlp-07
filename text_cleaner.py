import re
import pandas as pd

class TextCleaner:
    """
    TextCleaner 클래스는 텍스트 데이터를 정리하는데 사용되는 유틸리티 클래스입니다.
    이 클래스는 텍스트에서 의미 없는 영어 단어와 숫자, 특수 문자를 제거하고,
    깨끗한 텍스트를 반환하는 데 중점을 둡니다.
    """

    # 의미 있는 영단어를 찾기 위한 정규 표현식 (두 글자 이상의 대문자 단어 또는 'vs'만 남깁니다)
    meaningful_pattern = r'[A-Z]{2,}|vs'
    
    # 숫자 뒤에 나타나는 특수 문자 패턴을 찾기 위한 정규 표현식
    special_char_pattern = r'(?<![.\u2026∼])([^\w\s.\u2026∼])\d+'

    @staticmethod
    def remove_eng_noise(text):
        """
        텍스트에서 의미 없는 영어 노이즈 단어를 제거하는 메서드.
        한글을 포함하는 텍스트에서 의미 없는 영어 단어를 제거하고,
        텍스트의 흐름을 자연스럽게 유지하는 방식으로 클리닝합니다.
        
        :param text: 텍스트 문자열
        :return: 노이즈가 제거된 텍스트 문자열
        """
        cleaned_text = []  # 정리된 텍스트를 담을 리스트
        current_word = ''  # 현재 단어를 담을 변수

        for char in text:
            if re.match(r'[\uAC00-\uD7A3]', char):  # 한글인 경우
                if current_word:  # 이전에 저장된 영어 단어가 있으면
                    if re.match(TextCleaner.meaningful_pattern, current_word):  # 의미 있는 단어인지 확인
                        cleaned_text.append(current_word)  # 의미 있는 단어 추가
                    current_word = ''  # 단어 초기화
                cleaned_text.append(char)  # 한글 추가
            elif char.isalpha():  # 영어 알파벳인 경우
                current_word += char  # 단어에 추가
            else:  # 그 외 문자 (특수 문자 등)
                if current_word:  # 영어 단어가 끝나면
                    if re.match(TextCleaner.meaningful_pattern, current_word):  # 의미 있는 단어인지 확인
                        cleaned_text.append(current_word)  # 의미 있는 단어 추가
                    current_word = ''  # 단어 초기화
                cleaned_text.append(char)  # 특수 문자나 기타 문자 그대로 추가

        if current_word and re.match(TextCleaner.meaningful_pattern, current_word):  # 마지막에 남은 단어 처리
            cleaned_text.append(current_word)

        return ''.join(cleaned_text)  # 정리된 텍스트 반환

    @staticmethod
    def clean_number(text):
        """
        텍스트에서 숫자와 특정 특수 문자를 제거하는 메서드.
        숫자가 알파벳이나 한글 사이에 있는 경우에는 공백으로 대체하며,
        특수 문자는 정의된 패턴에 맞게 처리합니다.
        
        :param text: 텍스트 문자열
        :return: 숫자와 특수 문자가 클리닝된 텍스트 문자열
        """
        # 숫자가 영어 알파벳이나 한글 사이에 있을 때 공백으로 대체
        text = re.sub(r'(?<=[a-zA-Z가-힣])\d+(?=[a-zA-Z가-힣])', ' ', text)
        # 특수 문자를 정규식에 맞게 처리
        text = re.sub(TextCleaner.special_char_pattern, r'\1', text)
        # 여분의 공백 제거
        return re.sub(r'\s+', ' ', text).strip()  # 공백을 하나로 정리하고 양옆 공백 제거

# 사용 예시
if __name__ == "__main__":
    # 샘플 데이터프레임 생성
    data = {'text': ["This is NOISE 123 and → 화살표.", "Check AAPL vs 300 stocks."]}
    df = pd.DataFrame(data)
    
    # remove_eng_noise와 clean_number를 DataFrame에 적용
    df['text'] = df['text'].apply(TextCleaner.remove_eng_noise)  # 의미 없는 영어 노이즈 제거
    df['text'] = df['text'].apply(TextCleaner.clean_number)  # 숫자 및 특수 문자 클리닝
    
    print(df)
