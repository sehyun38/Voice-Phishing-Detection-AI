# 자바 설치 필요 (JDK 17 필요)
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
from konlpy.tag import Okt
from config import TOKENIZER_NAME, FILE_PATH, INPUT_FILE_PATH, SENTENCE_ENDINGS_FILE, KEYWORD_PATH, MAX_LENGTH
import re

# 1. 설정 값
OUTPUT_ENCODING = "utf-8"

MIN_TOKEN_COUNT = 20  # 필터링 적용 시 최소 토큰 수
MIN_WORD_COUNT = 8  # 필터링 적용 없이 최소 단어 수

TEXT_COLUMN = "transcript"
LABEL_COLUMN = "label"

USE_STOPWORD_FILTERING = True

okt = Okt()

# 2. CSV 로딩 함수 (인코딩에 따라 읽기)
def load_csv_with_encoding(path):
    try:
        return pd.read_csv(path, encoding='utf-8')
    except:
        # utf-8로 안되면 cp949 시도
        return pd.read_csv(path, encoding='cp949')


# 3. CSV에서 종결어미 리스트 로딩
#    CSV 파일의 "hint" 컬럼에 종결어미가 각 행에 하나씩 있다고 가정합니다.
def load_sentence_endings():
    try:
        return load_csv_with_encoding(SENTENCE_ENDINGS_FILE)['hint'].dropna().tolist()
    except Exception as e:
        print(f"Failed to load sentence endings: {e}")
        return []


# 전역 변수로 종결어미 리스트 할당
sentence_endings = load_sentence_endings()

# 4. 문장 분할 함수 (종결 어미 기준으로 텍스트를 분할)
def is_sentence_ending_combined(text_segment, okt):
    if any(text_segment.endswith(e) for e in sentence_endings):
        return True

    # Okt 기준 체크
    tokens = okt.pos(text_segment)

    if tokens:
        last_token, pos = tokens[-1]

        # Mecab은 종결 어미를 'EF'로 태깅함
        if pos == "EF":
            print(f"[EF] {last_token} / {pos}")
            return True

    return False

def split_text_preserve_context(text, max_len):
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

        # CSV 기준 + Okt 기준이 결합된 종결어미 체크
        if is_sentence_ending_combined(decoded, okt):
            chunks.append(decoded)
            start = end
            continue

        # 뒤에서부터 마지막 종결어미 위치 탐색
        split_index = None
        for i in range(end - 1, start - 1, -1):
            sub_segment = input_ids[start:i + 1]
            sub_decoded = tokenizer.decode(sub_segment, skip_special_tokens=True).strip()
            if is_sentence_ending_combined(sub_decoded, okt):
                split_index = i + 1  # 종결어미 토큰까지 포함
                break

        # 종결어미를 찾지 못했거나 분할 위치가 매우 짧으면 강제 분할
        if split_index is None or split_index == start:
            chunks.append(decoded)
            start = end
        else:
            chunk_ids = input_ids[start:split_index]
            chunk_decoded = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
            chunks.append(chunk_decoded)
            start = split_index

    return chunks

# 5. 문자 반복 정규화 함수 (예: '아아아' -> '아')
def normalize_repeated_letters(word):
    return re.sub(r'(.)\1+', r'\1', word)

# 6. 중복 단어 제거 (Okt를 활용한 간단한 예제)
def remove_repeated_words_with_okt(text, important_words=None):
    if important_words is None:
        important_words = set()
    else:
        important_words = set(important_words)

    words = text.strip().split()
    result = []
    prev_stem = None

    for word in words:
        # 중요한 단어(important_words)에 포함된 단어는 그대로 유지합니다.
        if word in important_words:
            result.append(word)
            prev_stem = None
            continue

        normalized_word = normalize_repeated_letters(word)
        morphs = okt.morphs(normalized_word)
        current_stem = morphs[0] if morphs else normalized_word

        # 이전 형태소와 달라야 결과에 추가
        if current_stem != prev_stem:
            result.append(word)
        prev_stem = current_stem

    return ' '.join(result)

# 7. 컨텍스트 기반 텍스트 필터링 함수
def context_based_filtering(text, use_filtering=True, stop_pos=None, important_words=None):
    if stop_pos is None:
        stop_pos = ["Punctuation"]
    if not use_filtering:
        return text, [], text
    if important_words is None:
        important_words = []

    morphs = okt.pos(text, stem=False)
    removed_tokens = set(word for word, pos in morphs if pos in stop_pos)
    cleaned_text = text
    for token in removed_tokens:
        cleaned_text = cleaned_text.replace(token, '')
    cleaned_text = remove_repeated_words_with_okt(cleaned_text, important_words=important_words)
    return cleaned_text.strip(), list(removed_tokens), text

# 8. 전처리 파이프라인 함수: CSV 데이터셋을 처리합니다.
def process_dataset():
    new_data = []
    removed_stopword_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row[TEXT_COLUMN]
        label = row[LABEL_COLUMN]

        # 첫 단계: 최대 토큰 수 기준으로 텍스트를 분할(종결어미 기반)
        chunks = split_text_preserve_context(text, MAX_LENGTH)
        final_chunks = []
        for chunk in chunks:
            final_chunks.extend(split_text_preserve_context(chunk, MAX_LENGTH))

        # 각 chunk별로 불필요한 토큰 제거 및 텍스트 정제 수행
        for chunk in final_chunks:
            filtered_text, removed_tokens, original_text = context_based_filtering(
                chunk,
                use_filtering=USE_STOPWORD_FILTERING,
                stop_pos=["Punctuation"],
                important_words=important_words
            )
            filtered_tokens = tokenizer.tokenize(filtered_text)
            removed_stopword_count += len(tokenizer.tokenize(chunk)) - len(filtered_tokens)

            # 중요 단어가 포함되거나, 최소 토큰 수를 충족하는 경우만 데이터셋에 포함
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


# 9. 실행 부분
if __name__ == '__main__':
    # 입력 파일 및 필요한 CSV 파일들을 읽습니다.
    df = load_csv_with_encoding(INPUT_FILE_PATH)
    important_words = set(load_csv_with_encoding(KEYWORD_PATH)['word'].dropna().tolist())

    # 종결어미 리스트는 이미 전역 변수 sentence_endings에 저장되어 있음.
    # Tokenizer 초기화
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

    processed_df = process_dataset()
    processed_df.to_csv(FILE_PATH, index=False, encoding=OUTPUT_ENCODING)

    print("File saved successfully.")
