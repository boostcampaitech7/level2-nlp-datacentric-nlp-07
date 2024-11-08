from dataclasses import dataclass


@dataclass
class NewsProcessorConfig:
    """뉴스 처리기 설정을 위한 설정 클래스"""
    model_name: str = "aya-expanse:latest"
    temperature: float = 0
    seed: int = 42
    sample_size: int = 40
    random_state: int = 1

BASIC_COLUMNS = ['ID', 'text', 'target']
ARROW_REPLACEMENTS = {'→': '에서', '↑': '상승', '↓': '하락', '↔': ' '}
SPECIAL_CHAR_PATTERN = r'(?<!\d)\.(?!\d)|(?<!\d)%|[^가-힣A-Z\u4E00-\u9FFF\s0-9\.%㎜㎡]'


