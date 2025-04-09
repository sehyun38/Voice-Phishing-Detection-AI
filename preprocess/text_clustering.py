import os
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
import torch

# --- 환경변수 설정 (Windows MKL 메모리 이슈 완화용) ---
os.environ["OMP_NUM_THREADS"] = "8"

# --- 사용자 설정 변수 ---
MODEL_NAME = "BM-K/KoSimCSE-roberta"
CLUSTER_COUNT = 3
REMOVE_DUPLICATES = False  # 개선: 중복 제거 대신 표시만
VERBOSE = True
INPUT_FILE = "../보류/Interactive_VP_Dataset_exaone_360_v1.csv"
OUTPUT_FILE = "../dataset/Interactive_Dataset/vp_clustered_exaone_360_cleaned.csv"
IMPORTANT_KEYWORDS_FILE = "../dataset/phishing_words.csv"
SIMILARITY_THRESHOLD_LOW = 0.8
SIMILARITY_THRESHOLD_HIGH = 0.9
MIN_TOKEN_LENGTH = 10

# --- 로그 출력 ---
def log(msg, level="INFO", force=False):
    if VERBOSE or force:
        print(f"[{level}] {msg}")

# --- 데이터 로딩 ---
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding="utf-8", header=0)
    except:
        df = pd.read_csv(file_path, encoding="cp949", header=0)

    df.columns = df.columns.str.strip()

    log(f"✅ 전체 데이터 로드 완료: {len(df)}개")
    log(f"📂 컬럼 목록: {df.columns.tolist()}")
    log(f"📊 컬럼별 dtype:\n{df.dtypes}")

    return df

# --- 중요 단어 로드 ---
def load_important_keywords(path):
    try:
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except:
            df = pd.read_csv(path, encoding="cp949")
        keywords = df['word'].dropna().astype(str).tolist()
        log(f"중요 키워드 {len(keywords)}개 로드됨: {keywords[:10]}...")
        return keywords
    except Exception as e:
        log(f"중요 단어 파일 로드 실패: {e}", level="ERROR", force=True)
        return []

# --- 짧은 문장 필터링 ---
def filter_short_texts(df, important_keywords, min_token=10):
    def keep_row(row):
        if row["token_count"] >= min_token:
            return True
        for keyword in important_keywords:
            if keyword in row["transcript"]:
                return True
        return False

    before = len(df)
    df_filtered = df[df.apply(keep_row, axis=1)].reset_index(drop=True)
    log(f"최소 {min_token}토큰 필터 적용: {before} → {len(df_filtered)}개 유지")
    return df_filtered

# --- 문장 임베딩 ---
def get_embeddings(model_name, texts):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    embeddings = []
    skipped = 0
    for idx, text in enumerate(tqdm(texts, desc="임베딩 생성", unit="문장")):
        if not isinstance(text, str) or len(text.strip()) < 3:
            log(f"건너뜀 (too short): idx={idx}, text='{text}'", level="WARN")
            skipped += 1
            continue

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        if inputs['input_ids'].shape[1] < 3:
            log(f"건너뜀 (token shortage): idx={idx}, text='{text}'", level="WARN")
            skipped += 1
            continue

        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze()
            embeddings.append(emb.numpy())

    log(f"임베딩 완료: {len(embeddings)}개 / 스킵: {skipped}개")
    return embeddings

# --- 클러스터링 ---
def apply_clustering(embeddings, n_clusters):
    log(f"KMeans 클러스터링 (개수: {n_clusters})")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    log("Clustering done")
    return kmeans.labels_

# --- 중복 문장 처리 ---
def mark_duplicates(df, remove_duplicates=False,
                    similarity_threshold_low=SIMILARITY_THRESHOLD_LOW,
                    similarity_threshold_high=SIMILARITY_THRESHOLD_HIGH):
    embeddings = get_embeddings(MODEL_NAME, df['transcript'].tolist())
    sim_matrix = cosine_similarity(embeddings)

    is_duplicate = np.zeros(len(df), dtype=bool)
    max_similarities = np.zeros(len(df))
    most_similar_indices = np.full(len(df), -1)

    for i in tqdm(range(len(df)), desc="중복 문장 비교", unit="문장"):
        for j in range(len(df)):
            if i == j:
                continue
            sim = sim_matrix[i][j]
            if sim > max_similarities[i]:
                max_similarities[i] = sim
                most_similar_indices[i] = j
            if sim >= similarity_threshold_high:
                is_duplicate[j] = True
                log(f"{round(sim*100, 2)}% 유사 → 문장 {i+1}과 {j+1}", level="DUPLICATE")
            elif sim >= similarity_threshold_low:
                is_duplicate[j] = True
                log(f"{round(sim*100, 2)}% 유사 → 문장 {i+1}과 {j+1}", level="SIMILAR")

    df['is_duplicate'] = is_duplicate
    df['most_similar_id'] = df.iloc[most_similar_indices.astype(int)]['id'].values
    df['max_similarity'] = np.round(max_similarities * 100, 2)

    return df

# --- 클러스터링 전체 실행 ---
def cluster_and_save(df, model_name, n_clusters, output_file, remove_duplicates):
    df = df.copy()
    df = df.reset_index(drop=False).rename(columns={"index": "original_index"})

    log(f"클러스터링 대상: {len(df)}")
    embeddings = get_embeddings(model_name, df['transcript'].tolist())
    cluster_labels = apply_clustering(embeddings, n_clusters)
    df['cluster'] = cluster_labels + 1
    df = mark_duplicates(df, remove_duplicates)

    if output_file:
        df.to_csv(output_file, index=False, encoding="utf-8")
        log(f"결과 저장 → {output_file}", force=True)

    return df[["original_index", "cluster", "is_duplicate", "most_similar_id", "max_similarity"]]

# --- 메인 실행 ---
def main():
    df = load_data(INPUT_FILE)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["token_count"] = pd.to_numeric(df["token_count"], errors="coerce")

    important_keywords = load_important_keywords(IMPORTANT_KEYWORDS_FILE)

    phishing_df = df[df["label"] == 1].reset_index(drop=True)
    log(f"label==1 데이터: {len(phishing_df)}")

    phishing_df_filtered = filter_short_texts(phishing_df, important_keywords, min_token=MIN_TOKEN_LENGTH)
    clustered_df = cluster_and_save(phishing_df_filtered, MODEL_NAME, CLUSTER_COUNT, None, REMOVE_DUPLICATES)

    # 원본에 변경 적용
    df["cluster"] = None
    df["is_duplicate"] = False
    df["most_similar_id"] = None
    df["max_similarity"] = None

    for _, row in clustered_df.iterrows():
        idx = row["original_index"]
        df.at[idx, "cluster"] = row["cluster"]
        df.at[idx, "is_duplicate"] = row["is_duplicate"]
        df.at[idx, "most_similar_id"] = row["most_similar_id"]
        df.at[idx, "max_similarity"] = row["max_similarity"]

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    log(f"최종 결과 저장 완료 → {OUTPUT_FILE}", force=True)

    # --- 클러스터별 개수 출력 ---
    if "cluster" in df.columns:
        log("클러스터별 문장 수:", force=True)
        cluster_counts = df[df["label"] == 1]["cluster"].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            log(f"  ▶ Cluster {int(cluster_id)}: {count}개", force=True)

        # 미분류된 label==1 문장 확인
        unclustered = df[(df["label"] == 1) & (df["cluster"].isna())]
        log(f"❗ 클러스터에 포함되지 않은 label==1 문장 수: {len(unclustered)}개", force=True)

if __name__ == "__main__":
    main()
