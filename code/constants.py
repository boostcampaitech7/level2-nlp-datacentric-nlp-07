import os

SEED = 456
BASE_DIR = os.path.dirname(os.getcwd())
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

MODEL_NAME = 'klue/bert-base'
METRICS = 'f1'

TEST_SPLIT_SIZE = 0.3