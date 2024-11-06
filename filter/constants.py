# 매직 넘버 및 리터럴 문자열 상수 정의
ASCII_RATIO_THRESHOLD = 0.15
ENGLISH_COUNT_THRESHOLD = 2
SPECIAL_CHAR_RATIO_THRESHOLD = 0.042
CONTINUOUS_ENGLISH_LENGTH = 3
PERCENT_PATTERN = r'\d+\.\d+%'
SPECIAL_CHAR_PATTERN = r'(?<!\d)\.(?!\d)|(?<!\d)%|[^가-힣A-Z\u4E00-\u9FFF\s0-9\.%㎜㎡]'
ARROWS = ['→', '←', '↑', '↓', '↔']

# 컬럼명에 대한 상수 정의
COLUMN_TEXT = 'text'
COLUMN_LENGTH = 'length'
COLUMN_ASCII_COUNT = 'ascii_count'
COLUMN_IS_ENGLISH_ONLY = 'is_english_only'
COLUMN_IS_ALL_UPPERCASE = 'is_all_uppercase'
COLUMN_ASCII_RATIO = 'ascii_ratio'
COLUMN_ASCII_UPPERCASE_RATIO = 'ascii_uppercase_ratio'
COLUMN_ENGLISH_UPPERCASE_RATIO = 'english_uppercase_ratio'
COLUMN_ENGLISH_COUNT = 'english_count'
COLUMN_SPECIAL_CHAR_COUNT = 'special_char_count'
COLUMN_SPECIAL_CHAR_RATIO = 'special_char_ratio'
COLUMN_CONTINUOUS_ENGLISH = 'continuous_english'
