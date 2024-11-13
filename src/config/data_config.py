from dataclasses import dataclass


@dataclass
class NewsProcessorConfig:
    """뉴스 처리기 설정을 위한 설정 클래스
    Attributes:
        model_name (str): 사용할 AI 모델의 이름 (기본값: "aya-expanse:latest")
        temperature (float): 모델의 출력 다양성을 조절하는 파라미터 (0: 결정적, 1: 무작위적) (기본값: 0)
        seed (int): 재현성을 위한 랜덤 시드값 (기본값: 42)
        sample_size (int): 처리할 샘플의 크기 (기본값: 40)
        random_state (int): 랜덤 샘플링을 위한 시드값 (기본값: 1)
    """
    model_name: str = "aya-expanse:latest"
    temperature: float = 0
    seed: int = 42
    sample_size: int = 40
    random_state: int = 1

BASIC_COLUMNS = ['ID', 'text', 'target']
ARROW_REPLACEMENTS = {'→': '에서', '↑': '상승', '↓': '하락', '↔': ' '}
SPECIAL_CHAR_PATTERN = r'(?<!\d)\.(?!\d)|(?<!\d)%|[^가-힣A-Z\u4E00-\u9FFF\s0-9\.%㎜㎡]'


