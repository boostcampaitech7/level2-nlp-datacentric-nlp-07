# 모듈 설명

## 모듈 구조 및 주의 사항
- `noise_filter.py` 파일 내부에 `NoiseFilter` 추상 클래스가 있습니다.
- 새로운 노이즈 필터를 적용하려면 `NoiseFilter`를 상속 받아 `NoiseFilter.filter_noise` 메소드 함수 내에 구현하세요 :)
- 새로운 노이즈 필터를 구현 후 `__init__.py`에 다음과 같이 적용하세요
```python
from .{새로운_노이즈_필터_파일명}.py import {새로운_노이즈_필터_클래스명}
```
- 새로운 값(매직 넘버 혹은 리터럴 스트링)을 사용하는 경우 `constants.py`에 선언 후 import 해서 사용하세요.

## 모듈 사용 예시

```python
import pandas as pd
from filter import AsciiFilter

df = pd.read_csv('data/train.csv')
ascii_filter = AsciiFilter()
ascii_filtered_df = ascii_filter.filter_noise(df)
```