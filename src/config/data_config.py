BASIC_COLUMNS = ['ID', 'text', 'target']
ARROW_REPLACEMENTS = {'→': '에서', '↑': '상승', '↓': '하락', '↔': ' '}
SPECIAL_CHAR_PATTERN = r'(?<!\d)\.(?!\d)|(?<!\d)%|[^가-힣A-Z\u4E00-\u9FFF\s0-9\.%㎜㎡]'