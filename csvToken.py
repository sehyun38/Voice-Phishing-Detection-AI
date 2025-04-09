import pandas as pd
from transformers import AutoTokenizer

# KoBERT와 Exaone-3.5 토크나이저 불러오기
kobert_tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
exaone_tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct", trust_remote_code=True)
koGPT_tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2", trust_remote_code=True)
kluebert_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base",tokenizer_type=True)

def add_token_count_to_csv(input_csv, output_csv):
    # CSV 파일 읽기
    df = pd.read_csv(input_csv, encoding='utf-8')

    # 'transcript' 열이 존재하는지 확인
    if 'transcript' not in df.columns:
        raise ValueError("CSV 파일에 'transcript' 열이 존재하지 않습니다.")

    # 각 토크나이저를 사용하여 토큰 수 계산
    df['kobert_token_count'] = df['transcript'].astype(str).apply(lambda x: len(kobert_tokenizer.tokenize(x)))
    df['exaone_token_count'] = df['transcript'].astype(str).apply(lambda x: len(exaone_tokenizer.tokenize(x)))
    df['koGPT_token_count'] = df['transcript'].astype(str).apply(lambda x: len(koGPT_tokenizer.tokenize(x)))
    df['kluebert_token_count'] = df['transcript'].astype(str).apply(lambda x: len(kluebert_tokenizer.tokenize(x)))

    # 새로운 CSV 파일로 저장
    df.to_csv(output_csv,encoding="utf-8" , index=False)
    print(f"새로운 CSV 파일이 생성되었습니다: {output_csv}")


# 사용 예시
input_csv = './dataset/Interactive_Dataset/Interactive_VP_Dataset.csv'  # 기존 CSV 파일 경로
output_csv = './dataset/Interactive_VP_Dataset_TokenCount.csv'  # 저장할 새로운 CSV 파일 경로
add_token_count_to_csv(input_csv, output_csv)
