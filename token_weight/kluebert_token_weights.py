import pandas as pd
import torch
import unicodedata
from tqdm import tqdm
from transformers import AutoTokenizer
import math

# 설정
transcript_csv = '../dataset/Interactive_Dataset/Interactive_VP_Dataset_kluebert_360_v1.csv'
keyword_csv = '../dataset/phishing_words.csv'
pt_save_path = '../token_weight/token_weights_kluebert.pt'
csv_save_path = '../token_weight/token_weights_kluebert.csv'
max_length = 360
DEBUG_MODE = False

# 토크나이저 (KLUE-BERT 기준)
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

# 정규화 함수
def normalize(text):
    return unicodedata.normalize("NFKC", str(text)).strip()

# 유사도 판단 함수
def match_score(keyword, snippet):
    match = sum(1 for k, s in zip(keyword, snippet) if k == s)
    return match / len(keyword) if keyword else 0

THRESHOLDS = {
    (0, 3): 1.0,     # 완전 일치
    (4, 5): 0.8,
    (6, 999): 0.7
}

# 적용 여부 판단
def should_apply_weight(keyword, snippet):
    kw_len = len(keyword)
    match_len = sum(1 for k, s in zip(keyword, snippet) if k == s)

    for (lo, hi), ratio in THRESHOLDS.items():
        if lo <= kw_len <= hi:
            return match_len >= math.ceil(kw_len * ratio)
    return False

# 데이터 로딩
try:
    df = pd.read_csv(transcript_csv, encoding='utf-8')
except:
    df = pd.read_csv(transcript_csv, encoding='cp949')
try:
    kw_df = pd.read_csv(keyword_csv, encoding='utf-8')
except:
    kw_df = pd.read_csv(keyword_csv, encoding='cp949')

# 키워드 딕셔너리 (띄어쓰기 제거 + 정규화)
weight_dict = {
    normalize(row['word']).replace(" ", ""): float(row['weight']) if not pd.isna(row['weight']) else 1.0
    for _, row in kw_df.iterrows()
}

# 저장 구조
token_weights_dict = {}
csv_rows = []

# 본격 처리
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating Token Weights"):
    text = normalize(row['transcript'])
    clean_text = text.replace(" ", "")
    char_weights = [1.0] * len(text)

    matched_keywords = []

    for keyword, weight in weight_dict.items():
        kw_clean = keyword.replace(" ", "")
        for i in range(len(clean_text) - len(kw_clean) + 1):
            snippet = clean_text[i:i+len(kw_clean)]
            if should_apply_weight(kw_clean, snippet):
                matched_keywords.append(keyword)
                for j in range(i, i+len(kw_clean)):
                    if j < len(char_weights):
                        char_weights[j] = max(char_weights[j], round(weight, 4))

    # 토크나이즈 + offset mapping
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].squeeze(0)
    offset_mapping = encoded['offset_mapping'].squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    token_weights = torch.ones_like(input_ids, dtype=torch.float)

    for i, (start, end) in enumerate(offset_mapping.tolist()):
        if start == end or end > len(char_weights):
            continue
        token_weights[i] = max(char_weights[start:end])

    # PAD 토큰은 가중치 0
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is not None:
        token_weights[input_ids == pad_token_id] = 0.0

    token_weights_dict[idx] = token_weights
    csv_rows.append({
        'index': idx,
        'tokens': ' '.join(tokens),
        'weights': ' '.join([str(round(w.item(), 4)) for w in token_weights])
    })

    # 디버깅 출력
    if DEBUG_MODE and idx < 3:
        print(f"\n[DEBUG] Index: {idx}")
        print(f"[TEXT] {text}")
        print(f"[MATCHED] {matched_keywords}")
        for i, (tok, w) in enumerate(zip(tokens, token_weights.tolist())):
            if tok not in ['[PAD]']:
                print(f"{i:03} | {repr(tok):<15} | weight: {w:.4f}")

# 저장
torch.save(token_weights_dict, pt_save_path)
print(f"[PT 저장 완료] → {pt_save_path}")

csv_df = pd.DataFrame(csv_rows)
csv_df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')
print(f"[CSV 저장 완료] → {csv_save_path}")
