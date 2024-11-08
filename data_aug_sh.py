import pandas as pd
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from tqdm import tqdm

class SHDataAugmentor:
    """
    SHDataAugmentor 클래스는 주어진 텍스트 목록에 대해 데이터 증강을 수행하는 클래스입니다.
    주어진 뉴스 기사 제목을 기반으로 유사한 제목을 생성하고, 그 결과를 클리닝하여 반환합니다.
    """

    def __init__(self, model_name='aya-expanse:8b', temperature=0.7, seed=42):
        """
        클래스 초기화 함수로, Ollama 모델을 로드하고, 증강 템플릿을 설정합니다.
        
        :param model_name: 사용할 모델 이름 (기본값: 'aya-expanse:8b')
        :param temperature: 모델의 온도 값 (기본값: 0.7)
        :param seed: 모델 초기화 시 사용할 시드 값 (기본값: 42)
        """
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
        """
        텍스트에서 불필요한 대괄호와 따옴표를 제거하고 공백을 정리하는 정적 메서드.
        
        :param text: 클리닝할 텍스트 문자열
        :return: 클리닝된 텍스트 문자열
        """
        return re.sub(r'[\[\]"]', '', text).strip()

    def augment_text(self, text_list):
        """
        주어진 텍스트 리스트에 대해 증강을 수행하는 메서드.
        각 텍스트에 대해 모델을 호출하여 유사한 제목을 생성하고, 생성된 제목을 클리닝하여 반환합니다.
        
        :param text_list: 증강할 텍스트 목록 (리스트 형태)
        :return: 증강된 제목들의 리스트
        """
        augmented_titles = []  # 증강된 제목을 저장할 리스트
        for text in tqdm(text_list, desc='증강 중...', total=len(text_list)):  # 진행 상황 표시
            response = self.chain.invoke({"Text": text})  # 모델을 사용하여 증강된 제목 생성
            cleaned_response = self.clean_text(response)  # 생성된 제목 클리닝
            augmented_titles.append(cleaned_response)  # 증강된 제목 리스트에 추가
        return augmented_titles

class SHTextProcessor:
    """
    SHTextProcessor 클래스는 증강된 데이터와 텍스트를 처리하고, 불필요한 부분을 제거하는 유틸리티 클래스입니다.
    """

    @staticmethod
    def process_text(text):
        """
        텍스트에서 필요한 부분만 추출하는 정적 메서드.
        주어진 텍스트에서 따옴표로 감싸진 텍스트를 추출하거나, 없으면 첫 번째 줄을 반환합니다.
        
        :param text: 처리할 텍스트
        :return: 처리된 텍스트
        """
        matches = re.findall(r"['\"](.*?)['\"]", text)  # 따옴표로 감싼 텍스트 추출
        return matches[1] if len(matches) > 1 else matches[0] if matches else text.split('\n')[0]

    @staticmethod
    def clean_augmented_data(data):
        """
        증강된 데이터를 클리닝하고 불필요한 부분을 제거하는 메서드.
        'augmented_title' 열에서 불필요한 문자열을 처리하고, 길이가 4글자 이하인 데이터를 제거합니다.
        
        :param data: DataFrame 객체 (증강된 데이터 포함)
        :return: 클리닝된 DataFrame 객체
        """
        # 'augmented_title' 열에서 불필요한 문자열을 제거하고 길이 4글자 이하인 행 제거
        data['augmented_title'] = data['augmented_title'].apply(SHTextProcessor.process_text)
        data['text'] = data['augmented_title'].str.replace('증강된 제목: ', '', regex=False)
        return data[data['text'].str.len() > 4]  # 길이가 4글자 이하인 행 제거

    @staticmethod
    def save_to_csv(df, file_path):
        """
        DataFrame을 CSV 파일로 저장하는 메서드.
        
        :param df: 저장할 DataFrame 객체
        :param file_path: 저장할 파일 경로
        """
        df.to_csv(file_path, index=False)  # DataFrame을 CSV 파일로 저장
        print(f"결과가 '{file_path}'에 저장되었습니다.")  # 저장 완료 메시지 출력

# 전체 워크플로우를 실행하는 클래스
class SHWorkflow:
    """
    SHWorkflow 클래스는 전체 데이터 증강 및 클리닝 파이프라인을 관리하는 클래스입니다.
    주어진 경로에서 데이터를 로드하고, 데이터 증강과 클리닝을 수행한 후 결과를 저장합니다.
    """

    def __init__(self, input_path, model_name='aya-expanse:8b'):
        """
        클래스 초기화 함수로, 입력 경로와 모델을 설정합니다.
        
        :param input_path: 입력 데이터 파일 경로
        :param model_name: 사용할 모델 이름 (기본값: 'aya-expanse:8b')
        """
        self.input_path = input_path  # 입력 데이터 경로
        self.augmentor = SHDataAugmentor(model_name=model_name)  # 데이터 증강기 초기화

    def run_augmentation_and_clean(self, output_path):
        """
        데이터 증강 및 클리닝을 수행하고 결과를 지정된 경로에 저장하는 메서드.
        
        :param output_path: 증강된 데이터를 저장할 출력 파일 경로
        """
        # 데이터 로드 및 증강 수행
        data = pd.read_csv(self.input_path)  # 입력 데이터 로드
        augmented_titles = self.augmentor.augment_text(data['text'].tolist())  # 텍스트 증강
        data['augmented_title'] = augmented_titles  # 증강된 제목을 DataFrame에 추가

        # 클리닝 후 결과를 저장
        cleaned_data = SHTextProcessor.clean_augmented_data(data)  # 클리닝 수행
        SHTextProcessor.save_to_csv(cleaned_data, output_path)  # 결과를 CSV로 저장

# 실행 예시
if __name__ == "__main__":
    # 경로 설정
    input_path = './data/noise.csv'  # 입력 데이터 경로
    cleaned_output_path = 'aug_sh.csv'  # 증강된 데이터를 저장할 출력 파일 경로
    
    # 워크플로우 실행
    workflow = SHWorkflow(input_path)  # SHWorkflow 객체 생성
    workflow.run_augmentation_and_clean(cleaned_output_path)  # 증강 및 클리닝 수행
