# Baseline Code
Baseline code를 모듈화

## 모듈 구조 및 주의 사항
- `main.py` 파일 내부에 `Model` 클래스가 있습니다.
- Model 객체를 생성해 토크나이저/모델등을 초기화 하고, `Model.train()`, `Model.evaluate()`를 통해 학습과 추론을 진행합니다.
- 추론 결과는  constant의 OUTPUT_DIR에 저장됩니다.

## 모듈 사용 예시

```python
basic_model = Model()

train_data = pd.read_csv(os.path.join(DATA_DIR, 'noise_moved.csv'))
dataset_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

basic_model.train(train_data)
basic_model.evaluate(dataset_test)
```