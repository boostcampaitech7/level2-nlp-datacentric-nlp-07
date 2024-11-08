import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from tqdm import tqdm
import re


data= pd.read_csv('./data/train.csv')

model = OllamaLLM(
    model='aya-expanse:8b',
    temperature=0,
    seed=42
)

def clean_text(text):
    # 대괄호와 따옴표 제거하고 앞뒤 공백 제거
    cleaned = re.sub(r'[\[\]"]', '', text).strip()
    return cleaned

template = """
    다음은 뉴스 기사 제목입니다.
    뉴스 기사 제목: {Text}
    지시사항:
    1. 뉴스 기사 제목을 읽어.
    2. 해당 제목를 {Field}로 번역해.
    지정 답변 형식: "[번역된 제목]"
    """
            
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
translate = []
for text in tqdm(data['text'].to_list(), desc='RTT....', total=len(data)):
    response = chain.invoke({"Text": text, "Field": '스페인어'})
    cleand =  clean_text(response)
    translate.append(cleand)
    
data['translate_ger'] = translate
data.to_csv('train_aug.csv', index=False)