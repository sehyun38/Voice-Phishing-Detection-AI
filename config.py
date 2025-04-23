import torch
from transformers import  AutoTokenizer

#토크나이저 설정
TOKENIZER_NAME = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
VOCAB_SIZE = len(tokenizer)

#디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 하이퍼 파라미터 설정
MAX_LENGTH = 100    # 최대 토큰 수
batch_size = 32         # 학습 배치사이즈
num_epochs = 100        # klue bert 제외 전체 에포크 수
num_classes = 2        # 출력층 수
n_splits = 5

#파일 경로
INPUT_FILE_PATH = "./dataset/Interactive_Dataset/Interactive_VP_Dataset.csv"
FILE_PATH =  './dataset/Interactive_Dataset/Interactive_VP_Dataset_kluebert_100_v1.csv'
KEYWORD_PATH = './dataset/phishing_words.csv'
PT_SAVE_PATH = './token_weight/token_weights_kluebert.pt'
CSV_SAVE_PATH = './token_weight/token_weights_kluebert.csv'
SENTENCE_ENDINGS_FILE = './dataset/grammar_data/sentence_endings.csv'

