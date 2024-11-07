# Data-Centric 주제 분류

## 📕 프로젝트 개요

- 부스트캠프 AI Tech `NLP` 트랙에서 진행된 Level 2 프로젝트
- `Data-Centric Text Classification` Task:
  - 주어진 뉴스 헤드라인을 7개의 카테고리로 분류하는 문제
  - 자연어 문장을 주제에 따라 정확하게 분류하여 모델의 언어 이해 능력을 평가
  - 학습 데이터는 2,800개이며, 테스트 데이터는 30,000개로 구성됨
- **카테고리**: 생활문화, 스포츠, 세계, 정치, 경제, IT과학, 사회

  - `ID`: 각 데이터 샘플의 고유번호
  - `text`: 뉴스 헤드라인 (한국어 텍스트, 일부 영어 및 한자 포함)
  - `target`: 정수로 인코딩된 라벨

- 평가 지표는 `Macro F1 Score`를 기준으로 함

## 📆 세부 일정

#### 공통 일정

- 프로젝트 기간 (2주): 10월 28일 (월) ~ 11월 7일 (목)
- 데이터 EDA 및 전처리: 10월 28일 (월) ~ 10월 30일 (수)
- 데이터 증강 및 정제: 10월 31일 (목) ~ 11월 2일 (토)
- 모델 학습 및 실험: 11월 3일 (일) ~ 11월 6일 (수)
- 최종 성능 평가 및 리포트 작성: 11월 7일 (목)

## 😁 팀 소개

<table style="width: 100%; text-align: center;">
  <tr>
    <th>강감찬</th>
    <th>이채호</th>
    <th>오승범</th>
    <th>이서현</th>
    <th>유채은</th>
    <th>서재덕</th>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;"><img src="./docs/README/강감찬.gif" alt="강감찬" width="100" height="100"></td>
    <td style="text-align: center; vertical-align: middle;"><img src="./docs/README/이채호.gif" alt="이채호" width="100" height="100"></td>
    <td style="text-align: center; vertical-align: middle;"><img src="./docs/README/오승범.gif" alt="오승범" width="100" height="100"></td>
    <td style="text-align: center; vertical-align: middle;"><img src="./docs/README/이서현.gif" alt="이서현" width="100" height="100"></td>
    <td style="text-align: center; vertical-align: middle;"><img src="./docs/README/유채은.gif" alt="유채은" width="100" height="100"></td>
    <td style="text-align: center; vertical-align: middle;"><img src="./docs/README/서재덕.gif" alt="서재덕" width="100" height="100"></td>
  </tr>
  <tr>
    <td><a href="https://github.com/gsgh3016">@강감찬</a></td>
    <td><a href="https://github.com/chell9999">@이채호</a></td>
    <td><a href="https://github.com/Sbeom12">@오승범</a></td>
    <td><a href="https://github.com/seohyeon0677">@이서현</a></td>
    <td><a href="https://github.com/canolayoo78">@유채은</a></td>
    <td><a href="https://github.com/jduck301">@서재덕</a></td>
  </tr>
  <tr>
    <td>데이터 클러스터링, 증강</td>
    <td>TODO: 채호 한 일 추가(주요 업무만)</td>
    <td>TODO: 승범 한 일 추가(주요 업무만)</td>
    <td>TODO: 서현 한 일 추가(주요 업무만)</td>
    <td>TODO: 채은 한 일 추가(주요 업무만)</td>
    <td>TODO: 재덕 한 일 추가(주요 업무만)</td>
  </tr>
</table>

## 프로젝트 수행 절차 및 방법

본 프로젝트는 데이터 중심 접근 방식을 활용하여 모델 성능을 개선하는 것을 목표로 하였습니다. 프로젝트는 다음과 같은 단계로 진행되었습니다:

### 1. 목표 설정 및 계획 수립

프로젝트 시작 시 각 팀원이 목표를 공유하고, 대회의 목적에 맞게 SMART 목표를 설정하였습니다. 리더보드 성적보다는 데이터 중심적 기법을 깊이 이해하는 데 중점을 두었으며, 각 팀원이 역할을 분담하여 협업을 효율적으로 진행할 수 있도록 계획을 수립하였습니다.

### 2. 데이터셋 EDA 및 전처리

#### 2.1 데이터셋 분석

학습 데이터의 구조를 파악하기 위해 EDA를 진행하였으며, 주제별 분포와 주요 특징을 분석하였습니다. 텍스트 내 언어 혼합 및 특수 문자 등을 파악하고, 데이터 정제의 필요성을 확인하였습니다.

#### 2.2 데이터 정제 및 전처리

불필요한 데이터를 필터링하고, 정제된 데이터를 모델 학습에 적합한 형태로 변환하였습니다. 이 과정에서 한글이 포함되지 않은 텍스트를 제거하고, 특수 문자를 처리하였습니다. 

또한, 데이터 증강 기법을 활용하여 다양한 데이터 변형을 시도하였습니다.

### 3. 모델 설계 및 구현

베이스라인 코드의 구조를 유지하면서 데이터 중심적 접근을 통해 성능을 개선하였습니다. 구체적인 방법은 다음과 같습니다:

#### 3.1 데이터 증강 및 필터링

- T5 모델을 활용하여 뉴스 헤드라인을 변형하거나 추가 생성
- 다양한 증강 기법을 실험하여 최적의 데이터 조합을 찾음
- Noise Filtering: 라벨링 오류 탐지 및 데이터 정제를 통한 성능 향상

#### 3.2 하이퍼파라미터 최적화

- 학습 효율성을 높이기 위해 `max_sequence_length` 및 `batch_size`를 조정
- 데이터 split 비율을 실험하여 최적의 학습-검증 비율을 설정

### 4. 모델 성능 실험 및 평가

#### 4.1 Macro F1 Score 계산

각 라벨별 F1 점수를 산출한 후 평균을 계산하여 모델의 성능을 평가하였습니다. 추가로 데이터 전처리와 증강이 성능에 미치는 영향을 체계적으로 분석하였습니다.

## 프로젝트 아키텍처

#### 데이터 전처리 모듈

<img src="./docs/image/README/preprocessing.png" alt="Data Preprocessing Module" width="600">

#### 증강 및 필터링 모듈

<img src="./docs/image/README/augmentation.png" alt="Data Augmentation Module" width="600">

## 프로젝트 결과

|           |  Public  | Private  |
| :-------: | :------: | :------: |
|   정확도  | 예시 텍스트 입니다. | 예시 텍스트 입니다. |
| F1 점수   | 예시 텍스트 입니다. | 예시 텍스트 입니다. |
| 최종 등수 | 예시 텍스트 입니다. | 예시 텍스트 입니다. |

## Getting Started

- main.py를 실행하여 프로젝트를 시작할 수 있습니다:
```bash
python main.py
```

- 데이터 경로 및 설정은 'src/config/path_config.py'에서 관리됩니다.
```python
data_path = "./data/"
train_data = "./data/train.csv"
test_data = "./data/test.csv"
outputs = "./outputs/"
```

- 하이퍼파라미터 및 모델 설정은 'src/utils/arguments.py'에서 조정 가능합니다.

# Appendix

## A. 주요 기술 스택 및 도구

- **Python**: 데이터 처리 및 모델 구현
- **Hugging Face Transformers**: NLP 모델 활용
- **Scikit-learn**: 데이터 전처리 및 분석
- **PyTorch**: 모델 학습 및 튜닝
- **Jupyter Notebook**: 데이터 분석 및 시각화
- **LangChain**: 데이터 증강 실험

## B. 실험 환경 및 하드웨어

- **GPU (예: NVIDIA V100)**: 모델 학습 및 실험에 사용
- **Server: V100**: SSH 프로토콜로 접근하는

## C. 프로젝트 구조

```text
├── data
│   ├── train.csv
│   └── test.csv
├── docs
│   ├── image
│   │   ├── README
│   │   │   ├── augmentation.png
│   │   │   ├── preprocessing.png
│   │   │   └── team.png
├── src
│   ├

── config
│   │   └── path_config.py
│   ├── main.py
│   ├── preprocessing
│   │   ├── data_cleaning.py
│   │   ├── data_filtering.py
│   │   └── data_augmentation.py
│   └── utils
│       ├── arguments.py
│       ├── metrics.py
│       └── helper_functions.py
└── outputs
```

## D. 실험 결과

| 모델                    | 정확도 | F1 Score |
| ----------------------- | ------ | -------- |
| T5 증강 데이터          | 예시 텍스트 입니다. | 예시 텍스트 입니다. |
| 필터링 데이터           | 예시 텍스트 입니다. | 예시 텍스트 입니다. |

## E. 추가 참고 문헌

- **T5 모델**: Paper on "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
- **BM25 알고리즘**: Information Retrieval Journal
- **Hugging Face**: [Hugging Face Documentation](https://huggingface.co/docs)

---

추가 내용이나 수정할 부분이 있으면 알려주세요!