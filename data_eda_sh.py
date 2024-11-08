import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

df = pd.read_csv('../data/train.csv')

# 공백으로 대체
df['text'] = df['text'].str.replace('…', ' ', regex=False)
df['text'] = df['text'].str.replace('...', ' ', regex=False)
df['text'] = df['text'].str.replace('·', ' ', regex=False)
# 화살표 기호 의미 대체
df['text'] = df['text'].str.replace('→', '에서', regex=False)
df['text'] = df['text'].str.replace('↑', '상승', regex=False)
df['text'] = df['text'].str.replace('↓', '하락', regex=False)
df['text'] = df['text'].str.replace('↔', ' ', regex=False)

# 특수 기호의 패턴 정의
# 이 예시에서는 영숫자 이외의 모든 기호를 특수 기호로 간주합니다.
# special_char_pattern = r'[^가-힣A-Z\u4E00-\u9FFF\s0-9]'
# .숫자% 형식을 제외하고, ㎜도 특수 기호에서 제외
special_char_pattern = r'(?<!\d)\.(?!\d)|(?<!\d)%|[^가-힣A-Z\u4E00-\u9FFF\s0-9\.%㎜㎡]'

# 각 text에 포함된 특수 기호의 개수를 세는 함수 정의
def count_special_characters(text):
    return len(re.findall(special_char_pattern, text))

# 데이터프레임에 새로운 열 추가
df['special_char_count'] = df['text'].apply(count_special_characters)
df['special_char_ratio'] = df['special_char_count'] / df['text'].str.len()

df_sorted = df.sort_values(by='special_char_ratio', ascending=False)

# special_char_ratio가 0.2 이상인 데이터 필터링
df_high_ratio = df_sorted[df_sorted['special_char_ratio'] >= 0.042]


# df_high_ratio를 CSV 파일로 저장하기
df_high_ratio.to_csv('../data/df_high_ratio.csv', index=False, encoding='utf-8-sig')


# df_high_ratio의 인덱스를 이용하여 제외할 행을 찾음
df_excluded = df[~df.index.isin(df_high_ratio.index)]

# 결과를 CSV 파일로 저장
df_excluded.to_csv('../data/noise_free.csv', index=False)