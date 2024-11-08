from pathlib import Path 


BASE = Path(__file__).resolve().parent.parent.parent
data_path = "./data/train.csv"
ascii_path1 = "./data/noise_scs.csv"
ascii_path2 = "./data/noise_s.csv"
data_output = "./data"


DATA_PATH = Path(BASE, data_path)
DATA_OUTPUT = Path(BASE, data_output)
ASCII_PATH_SCS = Path(BASE, ascii_path1)
ASCII_PATH_S = Path(BASE, ascii_path2)