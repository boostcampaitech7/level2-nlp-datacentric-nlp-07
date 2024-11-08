import pandas as pd
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from tqdm import tqdm

class SHDataAugmentor:
    def __init__(self, model_name='aya-expanse:8b', temperature=0.7, seed=42):
        # 모델 초기화
        self.model = OllamaLLM(model=model_name, temperature=temperature, seed=seed)
        
        # 데이터 증강 템플릿 설정
        self.template = """
            다음은 뉴스 기사 제목입니다.
            뉴스 기사 제목: "{Text}"
            지시사항:
            1. 주어진 뉴스 기사 제목을 읽어라.
            2. 해당 제목을 바탕으로 유사한 제목을 생성하라.
            지정 답변 형식: "[증강된 제목]"
        """
        
        # 템플릿과 모델을 연결한 체인 생성
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.chain = self.prompt | self.model

    @staticmethod
    def clean_text(text):
        """텍스트에서 불필요한 대괄호와 따옴표를 제거하고 공백을 정리."""
        return re.sub(r'[\[\]"]', '', text).strip()

    def augment_text(self, text_list):
        """주어진 텍스트 리스트를 증강하여 반환."""
        augmented_titles = []
        for text in tqdm(text_list, desc='증강 중...', total=len(text_list)):
            response = self.chain.invoke({"Text": text})
            cleaned_response = self.clean_text(response)
            augmented_titles.append(cleaned_response)
        return augmented_titles

class SHTextProcessor:
    @staticmethod
    def process_text(text):
        """텍스트에서 필요한 부분만 추출."""
        matches = re.findall(r"['\"](.*?)['\"]", text)
        return matches[1] if len(matches) > 1 else matches[0] if matches else text.split('\n')[0]

    @staticmethod
    def clean_augmented_data(data):
        """증강된 데이터를 클리닝하고 불필요한 부분을 제거."""
        # 'augmented_title' 열에서 불필요한 문자열을 제거하고 길이 4글자 이하인 행 제거
        data['augmented_title'] = data['augmented_title'].apply(SHTextProcessor.process_text)
        data['text'] = data['augmented_title'].str.replace('증강된 제목: ', '', regex=False)
        return data[data['text'].str.len() > 4]

    @staticmethod
    def save_to_csv(df, file_path):
        """DataFrame을 CSV 파일로 저장."""
        df.to_csv(file_path, index=False)
        print(f"결과가 '{file_path}'에 저장되었습니다.")

# 전체 워크플로우를 실행하는 클래스
class SHWorkflow:
    def __init__(self, input_path, model_name='aya-expanse:8b'):
        self.input_path = input_path
        self.augmentor = SHDataAugmentor(model_name=model_name)

    def run_augmentation_and_clean(self, output_path):
        """데이터 증강 및 클리닝을 수행하고 결과를 저장."""
        # 데이터 로드 및 증강 수행
        data = pd.read_csv(self.input_path)
        augmented_titles = self.augmentor.augment_text(data['text'].tolist())
        data['augmented_title'] = augmented_titles

        # 클리닝 후 결과를 저장
        cleaned_data = SHTextProcessor.clean_augmented_data(data)
        SHTextProcessor.save_to_csv(cleaned_data, output_path)

# 실행 예시
if __name__ == "__main__":
    # 경로 설정
    input_path = './data/noise.csv'
    cleaned_output_path = 'aug_sh.csv'
    
    # 워크플로우 실행
    workflow = SHWorkflow(input_path)
    workflow.run_augmentation_and_clean(cleaned_output_path)
