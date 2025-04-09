import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import re

# 설정 영역
INPUT_CSV_PATH = "../dataset/Interactive_Dataset/Interactive_VP_Dataset.csv"
OUTPUT_CSV_PATH = "../dataset/Interactive_Dataset/Interactive_VP_Dataset_kluebert_200_v1.csv"
OUTPUT_ENCODING = "utf-8"

# 조사/종결어미 파일 경로
PARTICLE_HINTS_FILE = '../dataset/grammar_data/particle_hints.csv'  # 조사 관련 CSV 파일 경로
SENTENCE_ENDINGS_FILE = '../dataset/grammar_data/sentence_endings.csv'  # 종결어미 관련 CSV 파일 경로

# 중요 단어가 저장된 파일 경로 ('word' 열을 사용)
IMPORTANT_WORDS_FILE = '../dataset/phishing_words.csv'

# 불용어 파일 경로 리스트
USE_STOPWORD_FILES = [
    # './stopwords/discourse_connectives.csv',
    # './stopwords/formal_closings.csv',
    # './stopwords/formal_expressions.csv',
    # './stopwords/response_fillers.csv',
    # './stopwords/Context-Aware_Korean_Stopwords__150_.csv,
]
USE_DEFAULT_STOPWORDS = True  # 기본 불용어 리스트 사용할지 여부

ENABLE_TOKEN_COUNT_FILTER = True  # 토큰 수 필터링 기능 사용 여부
MIN_TOKEN_COUNT = 15  # 필터링 기준이 되는 최소 토큰 수
MIN_WORD_COUNT = 6  # 필터링 기준이 되는 최소 단어 수

TOKENIZER_NAME = "klue/bert-base" # 사용할 Huggingface 토크나이저 이름

MAX_TOKEN_LENGTH = 200  # 최대 토큰 길이

TEXT_COLUMN = "transcript"  # 입력 CSV에서 텍스트가 있는 열 이름
LABEL_COLUMN = "label"  # 입력 CSV에서 라벨이 있는 열 이름

# 기본 불용어 리스트
DEFAULT_STOPWORDS = {
    "안녕하세요", "안녕", "반갑습니다", "수고하세요",
    "네", "예", "맞아요", "그래요", "그렇죠", "응", "엉",
    "아니요", "아니", "그건 아니고", "별로",
    "아", "어", "음", "음...", "그게", "저기", "흠", "아하", "오", "와",
    "그렇군요", "그래서요?", "그러네요", "그럼요", "그러게요", "그랬군요",
    "뭐", "그니까", "그러니까", "그러면", "근데", "아니 근데", "음 그래서",
    "아 네", "아 예", "아 그렇군요", "아 그래요?", "아 그렇죠", "네 네"
}

# 조사 불러오기
def load_particle_hints():
    try:
        hints_df = pd.read_csv(PARTICLE_HINTS_FILE, encoding='utf-8')
        return hints_df['hint'].dropna().tolist()
    except Exception as e:
        print(f"조사 파일 로딩 실패: {PARTICLE_HINTS_FILE} / {e}")
        return []

# 종결어미 불러오기
def load_sentence_endings():
    try:
        endings_df = pd.read_csv(SENTENCE_ENDINGS_FILE, encoding='utf-8')
        return endings_df['hint'].dropna().tolist()
    except Exception as e:
        print(f"종결어미 파일 로딩 실패: {SENTENCE_ENDINGS_FILE} / {e}")
        return []

# 불용어 불러오기 및 중요어 제외
def load_stopwords():
    stopwords = set()

    # 불용어 파일 불러오기
    for file in USE_STOPWORD_FILES:
        try:
            try:
                sw_df = pd.read_csv(file, encoding='utf-8')
            except:
                sw_df = pd.read_csv(file, encoding='cp949')
            stopwords.update(sw_df['word'].dropna().tolist())
        except Exception as e:
            print(f"불용어 파일 로딩 실패: {file} / {e}")

    if USE_DEFAULT_STOPWORDS:
        stopwords.update(DEFAULT_STOPWORDS)

    return stopwords

# 문장 자르기 기준 힌트 (조사 및 종결어미 포함)
def split_text_preserve_context(text, tokenizer, max_len, particle_hints, sentence_endings):
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    chunks = []
    start = 0
    while start < len(input_ids):
        end = min(start + max_len, len(input_ids))
        split_index = end
        for i in range(end, start + 10, -1):
            sub_text = tokenizer.decode(input_ids[start:i], skip_special_tokens=True)
            if any(sub_text.endswith(hint) for hint in particle_hints + sentence_endings):
                split_index = i
                break
        chunk = tokenizer.decode(input_ids[start:split_index], skip_special_tokens=True).strip()
        chunks.append(chunk)
        start = split_index
    return chunks

# 문맥 기반 불용어 제거
def context_based_filtering(text, stopwords, important_words):
    if any(keyword in text for keyword in important_words):
        return text  # 보이스피싱 관련 중요 단어가 있으면 불용어 제거 안 함

    # 중요 단어가 없으면 불용어 제거
    tokens = text.split()
    filtered_tokens = [tok for tok in tokens if tok not in stopwords]
    return ' '.join(filtered_tokens)

# 전체 전처리 로직
def process_dataset(df, tokenizer, stopwords, important_words, particle_hints, sentence_endings):
    new_data = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row[TEXT_COLUMN]
        label = row[LABEL_COLUMN]

        # 1. 토큰화 후 최대 토큰으로 자르기
        chunks = split_text_preserve_context(text, tokenizer, MAX_TOKEN_LENGTH, particle_hints, sentence_endings)

        # 2. 자른 각 문장에 대해 최소 토큰 수 기준으로 필터링
        for chunk in chunks:
            chunk_tokens = tokenizer.tokenize(chunk)  # chunk 토큰화
            filtered_tokens = [tok for tok in chunk_tokens if tok not in stopwords]  # 불용어 제거

            # 3. 중요 단어가 없으면 최소 토큰 수로 필터링하고, 중요 단어가 있으면 저장
            if any(important_word in chunk for important_word in important_words) or len(
                    filtered_tokens) >= MIN_TOKEN_COUNT:
                clean_text = tokenizer.convert_tokens_to_string(filtered_tokens).strip()  # 텍스트로 변환
                word_count = len(clean_text.split())  # 단어 수 계산

                # 4. 필터링된 텍스트가 최소 단어 수보다 작은 경우는 필터링 없이 바로 제거
                if word_count >= MIN_WORD_COUNT:
                    new_data.append({
                        'id': len(new_data) + 1,  # id는 new_data의 길이에 맞춰 자동으로 증가
                        TEXT_COLUMN: clean_text,
                        LABEL_COLUMN: label,
                        'token_count': len(filtered_tokens)  # 토큰 수로 저장
                    })

    return pd.DataFrame(new_data)

# 실행
try:
    df = pd.read_csv(INPUT_CSV_PATH, encoding='utf-8')
except:
    df = pd.read_csv(INPUT_CSV_PATH, encoding='cp949')

try:
    important_words = set(pd.read_csv(IMPORTANT_WORDS_FILE, encoding='utf-8')['word'].dropna().tolist())
except:
    important_words = set(pd.read_csv(IMPORTANT_WORDS_FILE, encoding='cp949')['word'].dropna().tolist())

particle_hints = load_particle_hints()
sentence_endings = load_sentence_endings()

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
stopwords = load_stopwords()  # 중요 단어와 불용어를 모두 불러옵니다.
processed_df = process_dataset(df, tokenizer, stopwords, important_words, particle_hints, sentence_endings)
processed_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding=OUTPUT_ENCODING)
print("파일 저장 완료")