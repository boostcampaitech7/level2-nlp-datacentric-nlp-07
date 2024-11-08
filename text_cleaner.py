import re
import pandas as pd

class TextCleaner:
    meaningful_pattern = r'[A-Z]{2,}|vs'
    special_char_pattern = r'(?<![.\u2026∼])([^\w\s.\u2026∼])\d+'

    @staticmethod
    def remove_eng_noise(text):
        """텍스트에서 영어 노이즈(의미 없는 영단어)를 제거."""
        cleaned_text = []
        current_word = ''

        for char in text:
            if re.match(r'[\uAC00-\uD7A3]', char):  # 한글인 경우
                if current_word:
                    if re.match(TextCleaner.meaningful_pattern, current_word):
                        cleaned_text.append(current_word)
                    current_word = ''  # current_word 초기화
                cleaned_text.append(char)  # 한글 추가
            elif char.isalpha():  # 영어 알파벳인 경우
                current_word += char  # 영어 문자를 current_word에 추가
            else:  # 기타 문자는 그대로 유지
                if current_word:
                    if re.match(TextCleaner.meaningful_pattern, current_word):
                        cleaned_text.append(current_word)
                    current_word = ''  # current_word 초기화
                cleaned_text.append(char)  # 기타 문자는 추가

        if current_word and re.match(TextCleaner.meaningful_pattern, current_word):
            cleaned_text.append(current_word)

        return ''.join(cleaned_text)

    @staticmethod
    def clean_number(text):
        """텍스트에서 숫자와 특정 특수 문자를 클리닝."""
        text = re.sub(r'(?<=[a-zA-Z가-힣])\d+(?=[a-zA-Z가-힣])', ' ', text)
        text = re.sub(TextCleaner.special_char_pattern, r'\1', text)
        return re.sub(r'\s+', ' ', text).strip()

# 사용 예시
if __name__ == "__main__":
    # 샘플 데이터프레임 생성
    data = {'text': ["This is NOISE 123 and → 화살표.", "Check AAPL vs 300 stocks."]}
    df = pd.DataFrame(data)
    
    # remove_eng_noise와 clean_number를 DataFrame에 적용
    df['text'] = df['text'].apply(TextCleaner.remove_eng_noise)
    df['text'] = df['text'].apply(TextCleaner.clean_number)
    
    print(df)
