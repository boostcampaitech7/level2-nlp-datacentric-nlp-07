from typing import List, Dict, Optional
import re 


class TextCleaner:
    """텍스트 전처리를 위한 클래스"""
    @staticmethod
    def extract_category(text: str) -> Optional[str]:
        """카테고리 추출"""
        pattern = r"뉴스 기사 분야 중 (.+?)(?:분야에|에)? 속합니다"
        match = re.search(pattern, text)
        return match.group(1) if match else text
    
    @staticmethod
    def clean_category(text: str) -> str:
        """카테고리 텍스트 정제"""
        return text.replace('*', '').strip()
    
    @staticmethod
    def clean_news(text: str) -> Optional[str]:
        """뉴스 텍스트에서 대괄호 내용 추출"""
        pattern = r'\[(.*?)\]'
        matches = re.search(pattern, text)
        return matches.group(1) if matches else None