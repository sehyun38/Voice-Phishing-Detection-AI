import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

# 설정
transcript_csv = '../dataset/Interactive_Dataset/Interactive_VP_Dataset_kobert_360_v3.csv'
keyword_csv = '../dataset/phishing_words.csv'
pt_save_path = 'token_weights_kobert.pt'
csv_save_path = 'token_weights_kobert.csv'
max_length = 360

# 토크나이저
tokenizer = AutoTokenizer.from_pretrained('monologg/kobert', trust_remote_code=True)

# 데이터 로딩
try:
    df = pd.read_csv(transcript_csv, encoding='utf-8')
except:
    df = pd.read_csv(transcript_csv, encoding='cp949')

try:
    kw_df = pd.read_csv(keyword_csv, encoding='utf-8')
except:
    kw_df = pd.read_csv(keyword_csv, encoding='cp949')

# 키워드 → 가중치 딕셔너리
weight_dict = {
    str(row['word']).strip(): float(row['weight']) if not pd.isna(row['weight']) else 1.0
    for _, row in kw_df.iterrows()
}

# 저장용
token_weights_dict = {}
csv_rows = []

# 본격 계산
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing transcripts"):
    text = str(row['transcript'])

    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids'].squeeze(0)
    token_weights = torch.ones_like(input_ids, dtype=torch.float)

    # 키워드 가중치 반영
    for word, weight in weight_dict.items():
        if word in text:
            word_tokens = tokenizer.tokenize(word)
            word_ids = tokenizer.convert_tokens_to_ids(word_tokens)
            word_ids_tensor = torch.tensor(word_ids)

            for i in range(len(input_ids) - len(word_ids) + 1):
                if torch.equal(input_ids[i:i + len(word_ids)], word_ids_tensor):
                    token_weights[i:i + len(word_ids)] *= round(weight, 4)

    # PAD 가중치 제거
    pad_token_id = tokenizer.pad_token_id or tokenizer.convert_tokens_to_ids("[PAD]")
    token_weights[input_ids == pad_token_id] = 0.0

    # 저장용 딕셔너리
    token_weights_dict[idx] = token_weights

    # CSV 저장용 (토큰 & 가중치)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    weights = token_weights.tolist()

    row_data = {
        'index': idx,
        'word': word,
        'tokens': ' '.join(tokens),
        'weights': ' '.join([str(round(w, 4)) for w in weights])
    }
    csv_rows.append(row_data)

# 저장
torch.save(token_weights_dict, pt_save_path)
print(f"[PT 저장 완료] → {pt_save_path}")

# CSV 저장
csv_df = pd.DataFrame(csv_rows)
csv_df.to_csv(csv_save_path, index=False, encoding='utf-8')
print(f"[CSV 저장 완료] → {csv_save_path}")
