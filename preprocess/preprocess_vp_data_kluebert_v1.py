#자바 설치 필요 jdk-17

import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
from konlpy.tag import Okt
from config import TOKENIZER_NAME, FILE_PATH, INPUT_FILE_PATH, SENTENCE_ENDINGS_FILE, KEYWORD_PATH, MAX_LENGTH

# 1. 설정
OUTPUT_ENCODING = "utf-8"

MIN_TOKEN_COUNT = 15        # 필터링 있는 최소 토큰 수
MIN_WORD_COUNT = 6          # 필터링 없는 최소 단어 길이 수

TEXT_COLUMN = "transcript"
LABEL_COLUMN = "label"

USE_STOPWORD_FILTERING = True

# 2. CSV 로딩 함수
def load_csv_with_encoding(path):
    try:
        return pd.read_csv(path, encoding='utf-8')
    except:
        return pd.read_csv(path, encoding='cp949')

# 3. 조사 및 종결 어미 리스트 로딩
def load_sentence_endings():
    try:
        return load_csv_with_encoding(SENTENCE_ENDINGS_FILE)['hint'].dropna().tolist()
    except Exception as e:
        print(f"Failed to load sentence endings: {e}")
        return []

# 4. 문장 분할 함수 (종결 어미 기준)
def split_text_preserve_context(text,  max_len):
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    if len(input_ids) <= max_len:
        return [tokenizer.decode(input_ids, skip_special_tokens=True).strip()]

    chunks = []
    start = 0
    while start < len(input_ids):
        end = min(start + max_len, len(input_ids))
        segment = input_ids[start:end]
        decoded = tokenizer.decode(segment, skip_special_tokens=True).strip()

        if any(decoded.endswith(e) for e in sentence_endings):
            chunks.append(decoded)
            start = end
            continue

        found = False
        for i in range(end - 1, start, -1):
            sub_segment = input_ids[start:i]
            sub_decoded = tokenizer.decode(sub_segment, skip_special_tokens=True).strip()
            if any(sub_decoded.endswith(e) for e in sentence_endings):
                chunks.append(sub_decoded)
                start = i
                found = True
                break

        if not found:
            fallback_chunk = tokenizer.decode(segment, skip_special_tokens=True).strip()
            chunks.append(fallback_chunk)
            start = end

    return chunks

# 5. 반복된 단어 제거
def remove_repeated_words_with_okt(text):
    okt = Okt()
    words = text.strip().split()
    result = []
    prev_stem = None
    for word in words:
        stemmed = okt.morphs(word)[0] if okt.morphs(word) else word
        if stemmed != prev_stem:
            result.append(word)
        prev_stem = stemmed
    return ' '.join(result)


# 6. 불용어 제거 및 반복어 제거 (조건부 처리)
def context_based_filtering(text, use_filtering=True,  stop_pos=None):
    if stop_pos is None:
        stop_pos = ["Punctuation"]

    if not use_filtering:
        return text, [], text

    okt = Okt()
    if important_words and any(word in text for word in important_words):
        return text, [], text

    morphs = okt.pos(text, stem=False)
    removed_tokens = set(word for word, pos in morphs if pos in stop_pos)
    cleaned_text = text
    for token in removed_tokens:
        cleaned_text = cleaned_text.replace(token, '')

    cleaned_text = remove_repeated_words_with_okt(cleaned_text)
    return cleaned_text.strip(), list(removed_tokens), text

# 7. 전처리 파이프 라인
def process_dataset():
    new_data = []
    removed_stopword_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row[TEXT_COLUMN]
        label = row[LABEL_COLUMN]

        chunks = split_text_preserve_context(text, MAX_LENGTH)
        final_chunks = []
        for chunk in chunks:
            final_chunks.extend(split_text_preserve_context(chunk, MAX_LENGTH))

        for chunk in final_chunks:
            filtered_text, removed_tokens, original_text = context_based_filtering(
                chunk,
                use_filtering=USE_STOPWORD_FILTERING,
                stop_pos=["Punctuation"],
            )
            filtered_tokens = tokenizer.tokenize(filtered_text)
            removed_stopword_count += len(tokenizer.tokenize(chunk)) - len(filtered_tokens)

            if any(word in filtered_text for word in important_words) or len(filtered_tokens) >= MIN_TOKEN_COUNT:
                word_count = len(filtered_text.split())
                if word_count >= MIN_WORD_COUNT:
                    new_data.append({
                        'id': len(new_data) + 1,
                        TEXT_COLUMN: filtered_text.strip(),
                        LABEL_COLUMN: label,
                        'token_count': len(filtered_tokens)
                    })

    print(f"Total removed stopword tokens: {removed_stopword_count}")
    return pd.DataFrame(new_data)

# 8. 실행
df = load_csv_with_encoding(INPUT_FILE_PATH)
important_words = set(load_csv_with_encoding(KEYWORD_PATH)['word'].dropna().tolist())
sentence_endings = load_sentence_endings()
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

processed_df = process_dataset()
processed_df.to_csv(FILE_PATH, index=False, encoding=OUTPUT_ENCODING)

print("File saved successfully.")
