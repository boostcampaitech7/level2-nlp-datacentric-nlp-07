import pandas as pd
from src.config.data_config import NewsProcessorConfig
from src.config.path_config import ASCII_PATH_SCS, CLEAN_ASCII
from src.utils.text_cleaner import TextCleaner
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from tqdm import tqdm
from typing import List, Dict, Optional

    
class NewsProcessor:
    """뉴스 처리를 위한 메인 클래스"""
    
    def __init__(self, config: NewsProcessorConfig):
        self.config = config
        self.cleaner = TextCleaner()
        
    def _create_model(self) -> OllamaLLM:
        """LLM 모델 생성"""
        return OllamaLLM(
            model=self.config.model_name,
            temperature=self.config.temperature,
            seed=self.config.seed
        )
        
    def load_data(self, input_path: str = ASCII_PATH_SCS) -> pd.DataFrame:
        """데이터 로드
        
        Args:
            input_path: 입력 데이터 파일 경로
            
        Returns:
            pd.DataFrame: 로드된 데이터
        """
        return pd.read_csv(input_path)
    
    
    def check_field(self, random_samples: pd.DataFrame) -> str:
        """뉴스 분야 확인"""
        model = self._create_model()
        docs = random_samples['text'].tolist()
        template = """
        다음은 특정 분야의 뉴스 기사 제목 50개입니다:
        문서 목록: {Docs}
        지시사항:
        1. 문서 목록들의 기사 제목들을 자세히 읽어보세요.
        2. 전체 기사 제목들이 공통적으로 어느 분야에 속하는지 판단.
        3. 해당 분야는 오직 한글로만 답변하고, 답변 형식을 지켜.
        지정 답변 형식: "뉴스 기사 분야 중 [분야명]에 속합니다."
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        response = chain.invoke({"Docs": docs})
        return self.cleaner.clean_category(self.cleaner.extract_category(response))
    
    def resolve_news(self, domain: str, data: pd.DataFrame, cond) -> List[str]:
        """뉴스 텍스트 해결"""
        model = self._create_model()
        resolve_data = []
        
        for text in tqdm(data[cond]['text'].tolist()[:1], desc=f"Processing {domain} news"):
            template = """
            다음은 {Domain} 분야의 뉴스 기사 제목입니다
            문서 목록: {Text}
            지시사항:
            1. {Domain} 분야를 바탕으로 {Text}의 뉴스 기사 제목을 생성.
            2. 만약 복구가 어렵다면 복구된 기사 제목은 패스로 해.
            3. 답변은 오직 한글로만 답변하고, 답변 형식을 지켜.
            지정 답변 형식: "[복구된 기사 제목]"
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | model
            response = chain.invoke({"Text": text, "Domain": domain})
            
            cleaned_news = self.cleaner.clean_news(response)
            if cleaned_news:
                resolve_data.append(cleaned_news)
                
        return resolve_data
    
    def process_data(self, data: pd.DataFrame, output_path: str) -> pd.DataFrame:
        """전체 데이터 처리 프로세스"""
        result_data = []
        
        for target in data['target'].unique().tolist():
            cond = data['target'] == target
            random_samples = data[cond].sample(
                n=self.config.sample_size, 
                random_state=self.config.random_state
            )
            
            print(f'현재 분류 정답 : {target}')
            response_target = self.check_field(random_samples)
            print('분류된 값:', response_target)
            
            clean_news_data = self.resolve_news(response_target, data, cond)
            
            for idx, text in enumerate(clean_news_data):
                result_data.append({
                    'ID': data[cond].iloc[idx]['ID'],
                    'target': target,
                    'text': text
                })
                
            print(f'{target} 분류 완료')
            print('------'*20)
        
        new_df = pd.DataFrame(result_data)
        new_df.to_csv(output_path, index=False)
        return new_df
    
    def save_data(self, df: pd.DataFrame, output_path: str) -> None:
        """처리된 데이터 저장
        
        Args:
            df: 저장할 데이터프레임
            output_path: 저장할 파일 경로
        """
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")

    def run(self) -> pd.DataFrame:
        """전체 뉴스 처리 프로세스 실행
        
        Args:
            input_path: 입력 데이터 파일 경로
            output_path: 출력 데이터 저장 경로
            
        Returns:
            pd.DataFrame: 처리된 뉴스 데이터
        """
        # 데이터 로드
        data = self.load_data(ASCII_PATH_SCS)
        
        # 데이터 처리
        processed_data = self.process_data(data, CLEAN_ASCII)
        
        print("정상화 완료!")
        return processed_data