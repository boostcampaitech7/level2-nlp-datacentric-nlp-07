# Data-Centric 주제 분류

## 📕 프로젝트 개요

### 대회 개요

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

### [대회 규칙](./docs/competition_rule.md)

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
    <td>데이터 및 라벨 오염 관련 실험, 데이터 증강</td>
    <td>TODO: 승범 한 일 추가(주요 업무만)</td>
    <td>TODO: 서현 한 일 추가(주요 업무만)</td>
    <td>TODO: 채은 한 일 추가(주요 업무만)</td>
    <td>TODO: 재덕 한 일 추가(주요 업무만)</td>
  </tr>
</table>

## 프로젝트 수행 절차 및 방법

본 프로젝트는 데이터 중심 접근 방식을 활용하여 모델 성능을 개선하는 것을 목표로 하였습니다. 프로젝트는 다음과 같은 단계로 진행되었습니다:

### 1. [데이터 분석](./docs/data_analysis.md)

[대회 규칙](./docs//competition.md) 기반으로 데이터 인스펙션(Data Inspection) 결과입니다.

### 2. [데이터 처리](./docs/data_processing.md)
#### 2.1 데이터 이상치 탐지

총 3가지 방법으로 데이터 이상치를 탐지했습니다.

1. ASCII 기반
2. 특수 문자 필터링 기반
3. 영문 필터링 기반

#### 2.2 데이터 정상화

텍스트 노이즈 정상화, 라벨 오류 정상화를 진행했습니다.

### 3. [데이터 증강](./docs/data_augmentation.md)

총 4가지 방법으로 데이터 증강을 실시했습니다.

1. BERT 기반 마스킹
2. LLM 활용 동의어 대체
3. *TODO: 추가*

### 4. 최종 순위

|           |  Public  | Private  |
| :-------: | :------: | :------: |
|   정확도  | 0.8400 | 0.8413 |
| F1 점수   | 0.8348 | 0.8366 |
| 최종 등수 | 12 | 11 |

## Getting Started

*TODO: 리팩토링 후 추가*

## 주요 기술 스택 및 도구

- **Python**: 데이터 처리 및 모델 구현
- **Hugging Face Transformers**: NLP 모델 활용
- **Scikit-learn**: 데이터 전처리 및 분석
- **PyTorch**: 모델 학습 및 튜닝
- **Jupyter Notebook**: 데이터 분석 및 시각화
- **LangChain**: 데이터 증강 실험
- *TODO: 프로젝트 진행 경과 정리 이후 실험에서 사용한 도구/기술 정리*

## 실험 환경 및 하드웨어

- **CPU**: intel xeon gold 5120
- **GPU**: NVIDIA V100
- **RAM**: 92GB
- **Storage**: 프로젝트 외부(환경) 20GB | 프로젝트 내부 100GB
- 서버 4대 SSH 접속하여 사용

## 참고 문헌

- **T5 모델**: Paper on "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
- **BM25 알고리즘**: Information Retrieval Journal
- **Hugging Face**: [Hugging Face Documentation](https://huggingface.co/docs)
