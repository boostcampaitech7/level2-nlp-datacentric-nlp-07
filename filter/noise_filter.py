import pandas as pd
from abc import ABC, abstractmethod


class NoiseFilter(ABC):
    """텍스트 데이터에서 노이즈를 탐지하는 추상 클래스"""

    @abstractmethod
    def filter_noise(self, data: pd.DataFrame) -> pd.DataFrame:
        """노이즈 필터링을 수행하는 메서드"""
        pass
