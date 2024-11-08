import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import re
from tqdm import tqdm

data= pd.read_csv('ascii.csv')

def extract_category(text: str):
    """카테고리 추출"""
    pattern = r"뉴스 기사 분야 중 (.+?)(?:분야에|에)? 속합니다"
    match = re.search(pattern, text)
    return match.group(1) if match else text

def clean_news(text: str):
    """뉴스 텍스트에서 대괄호 내용 추출"""
    bracket_pattern = r'\[(.*?)\]'
    bracket_match = re.search(bracket_pattern, text)
    if bracket_match:
        return bracket_match.group(1)
    
    asterisk_pattern = r'\*\*(.*?)\*\*'
    asterisk_match = re.search(asterisk_pattern, text)
    if asterisk_match:
        return asterisk_match.group(1)
    
    return text

model = OllamaLLM(
    model='qwen2.5:latest',
    temperature=0,
    seed=1004,
)
template = """
    다음은 특정 분야의 뉴스 기사 제목 40개입니다:
    기사 제목: {Docs}
    지시사항:
    1. 기사 제목들을 자세히 읽고, 해당 제목들이 공통적으로 뉴스 분야 큰 범주 중 어디에 속하는지 판단.
    2. 한글로, 답변 형식을 지켜서 대답해.
    답변 형식: "뉴스 기사 분야 중 [분야명]에 속합니다."
    """

pattern = r'\[(.*?)\]'

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

targets = []
for target in data['target'].unique().tolist():
    cond = data['target'] == target
    random_samples = data[cond].sample(n=50,random_state=15)
    docs = random_samples['text'].tolist()
    response = chain.invoke({"Docs": docs})
    targets.append(extract_category(response))

print(f'추론된 뉴스 분야: {targets}')

non_ascii = pd.read_csv('non_ascii.csv')
model_relabel = OllamaLLM(
    model='aya-expanse:8b',
    temperature=0,
    seed=1004,
)
template = """
    다음은 뉴스 기사 제목의 분야 및 제목입니다.
    뉴스 분야: {Docs}
    제목 : {Text}
    지시사항:
    1. 기사 제목을 읽고, 해당 제목이 제공된 뉴스 분야 중 어디에 속하는지 판단.
    2. 한글로, 답변 형식을 반드시 지켜.
    답변 형식: "[속한 분야명]"
    """
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model_relabel
results = []
for idx, row in tqdm(non_ascii.iterrows(), desc='Relabeling...', total=len(non_ascii)):
    text = row['text']
    response = chain.invoke({"Docs": targets, "Text": text})
    result = clean_news(response)
    temp = {
        'ID' : row['ID'],
        'text': text,
        'target' : result
    }
    results.append(temp)
    
check_field = pd.DataFrame(results)
check_field.to_csv('relabeled_data.csv', index=False)
