import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

# 설정 영역
INPUT_CSV_PATH = "../dataset/Interactive_Dataset/Interactive_VP_Dataset.csv"
OUTPUT_CSV_PATH = "../dataset/Interactive_Dataset/Interactive_VP_Dataset_kluebert_360_v1.csv"
OUTPUT_ENCODING = "utf-8"

# 조사/종결어미 파일 경로
PARTICLE_HINTS_FILE = '../dataset/grammar_data/particle_hints.csv'  # 조사 관련 CSV 파일 경로
SENTENCE_ENDINGS_FILE = '../dataset/grammar_data/sentence_endings.csv'  # 종결어미 관련 CSV 파일 경로

# 불용어 파일 경로 리스트
USE_STOPWORD_FILES = [
    # './stopwords/discourse_connectives.csv',
    # './stopwords/formal_closings.csv',
    # './stopwords/formal_expressions.csv',
    # './stopwords/response_fillers.csv',
    # './stopwords/Context-Aware_Korean_Stopwords__150_.csv,
]

USE_DEFAULT_STOPWORDS = True  # 기본 불용어 리스트 사용할지 여부
IMPORTANT_WORDS_FILE = '../dataset/phishing_words.csv'  # 중요 단어가 저장된 파일 경로 ('word' 열을 사용)

ENABLE_TOKEN_COUNT_FILTER = True  # 토큰 수 필터링 기능 사용 여부
MIN_TOKEN_COUNT = 5  # 필터링 기준이 되는 최소 토큰 수

ENABLE_BASIC_CLEANING = False  # 기본 텍스트 정제 사용 여부 (공백 정리 등)
REMOVE_SPECIAL_TOKENS = True  # 특수 토큰 ([CLS], [SEP] 등) 제거 여부

TOKENIZER_NAME = "klue/bert-base" # 사용할 Huggingface 토크나이저 이름

MAX_TOKEN_LENGTH = 360  # 최대 토큰 길이

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
    try:
        try:
            important_df = pd.read_csv(IMPORTANT_WORDS_FILE, encoding='utf-8')
        except:
            important_df = pd.read_csv(IMPORTANT_WORDS_FILE, encoding='cp949')
        important_words = set(important_df['word'].dropna().tolist())
        stopwords -= important_words
    except Exception as e:
        print(f"중요 단어 파일 로딩 실패: {e}")
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

def basic_cleaning(text):
    return ' '.join(text.split())

# 전체 전처리 로직
def process_dataset(df, tokenizer, stopwords, particle_hints, sentence_endings):
    new_data = []
    new_id = 1
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row[TEXT_COLUMN]
        label = row[LABEL_COLUMN]
        if ENABLE_BASIC_CLEANING:
            text = basic_cleaning(text)
        chunks = split_text_preserve_context(text, tokenizer, MAX_TOKEN_LENGTH, particle_hints, sentence_endings)
        for chunk in chunks:
            tokens = tokenizer.tokenize(chunk)
            filtered_tokens = [tok for tok in tokens if tok not in stopwords]
            if ENABLE_TOKEN_COUNT_FILTER and len(filtered_tokens) < MIN_TOKEN_COUNT:
                continue
            clean_text = tokenizer.convert_tokens_to_string(filtered_tokens).strip()
            new_data.append({
                'id': new_id,
                TEXT_COLUMN: clean_text,
                LABEL_COLUMN: label,
                'token_count': len(filtered_tokens)
            })
            new_id += 1
    return pd.DataFrame(new_data)

# 실행
try:
    df = pd.read_csv(INPUT_CSV_PATH, encoding='utf-8')
except:
    df = pd.read_csv(INPUT_CSV_PATH, encoding='cp949')

particle_hints = load_particle_hints()
sentence_endings = load_sentence_endings()

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
stopwords = load_stopwords()
processed_df = process_dataset(df, tokenizer, stopwords, particle_hints, sentence_endings)
processed_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding=OUTPUT_ENCODING)
print("파일 저장 완료")
