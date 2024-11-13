import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import re
from tqdm import tqdm


data_clean = pd.read_csv('clean_ascii.csv')
data = pd.read_csv('ascii.csv')
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

def extract_titles(text):
    titles = []
    lines = text.split('\n')
    
    for line in lines:
        # 숫자와 점으로 시작하는 라인에서 제목 추출
        if re.match(r'^\d+\.', line):
            # ** 또는 " 로 감싸진 제목 추출
            if '**' in line:
                title = re.search(r'\*\*(.*?)\*\*', line)
            else:
                title = re.search(r'"(.*?)"', line)
            
            if title:
                titles.append(title.group(1).strip())
    return titles
    

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
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

model_aug = OllamaLLM(
    model='aya-expanse:8b',
    temperature=0,
    seed=1004,
)

template_aug = """
    기사 분야: {Target}
    지시사항:
    1. 기사 분야을 기반으로 새로운 기사 제목 10개 생성해줘.
    답변 형식: " 1. 기사 제목1
                2. 기사 제목2
                3. 기사 제목3
    "
    """
pattern = r'\[(.*?)\]'
prompt = ChatPromptTemplate.from_template(template_aug)



targets = []
for target in data['target'].unique().tolist():
    cond = data['target'] == target
    random_samples = data[cond].sample(n=50,random_state=22)
    docs = random_samples['text'].tolist()
    response = chain.invoke({"Docs": docs})
    result = extract_category(response)
    targets.append(result)
    # 데이터 증강
print('필드:', targets)

results = []
for seed in tqdm(range(1, 100), desc="Generating...", total=100):
    model_aug = OllamaLLM(
        model='aya-expanse:8b',
        temperature=0,
        seed=seed,
    )
    chain2 = prompt | model_aug
    for target in targets:
        response_2 = chain2.invoke({"Target": target})
        titles = extract_titles(response_2)
        
        for title in titles:
            temp = {
            'target' : target,
            'generate_title' : title
            }
        results.append(temp)
    break
generate_df = pd.DataFrame(results)

generate_df.to_csv('generate.csv', index=False)