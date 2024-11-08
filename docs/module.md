# 모듈 구조

## 디렉토리 구조구조

```text
├── data
│   ├── train.csv
│   └── test.csv
├── docs
│   └── image
│       └── README
│           ├── augmentation.png
│           ├── preprocessing.png
│           └── team.png
├── src
│   ├── config
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

## 데이터 전처리 모듈

<img src="./docs/image/README/preprocessing.png" alt="Data Preprocessing Module" width="600">

## 증강 및 필터링 모듈

<img src="./docs/image/README/augmentation.png" alt="Data Augmentation Module" width="600">
