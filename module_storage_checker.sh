#!/bin/bash

# 1. 현재 사용 중인 pip의 경로 확인
pip_path=$(which pip)
echo "현재 사용 중인 pip의 경로: $pip_path"

# 2. pip가 가리키는 Python 실행 파일의 경로 확인
python_path=$(pyenv which python)
echo "pip가 가리키는 Python 실행 파일의 경로: $python_path"

# 3. site-packages 디렉토리 경로 확인
site_packages_dir=$($python_path -c "import site; print(site.getsitepackages()[0])")
echo "site-packages 디렉토리 경로: $site_packages_dir"

# 4. 서드 파티 모듈들의 용량 확인
echo "서드 파티 모듈들의 용량:"
du -sh $site_packages_dir/*

# 5. 서드 파티 모듈들의 총 용량 계산
total_size=$(du -sh $site_packages_dir | awk '{print $1}')

# 6. 총 용량을 빨간색으로 출력
RED='\033[0;31m'
NOCOLOR='\033[0m'
echo -e "${RED}서드 파티 모듈들의 총 용량: $total_size${NOCOLOR}"
