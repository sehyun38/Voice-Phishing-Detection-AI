import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
from my_models import BiLSTMAttention, TextCNN, RCNN, KLUEBertModel

# === 설정 ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_BERT_MODEL = True  # BERT 모델 포함 여부
EMBEDDING_DIM = 256
HIDDEN_DIM = 128
NUM_CLASSES = 2

# BERT tokenizer 및 vocab size
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base", trust_remote_code=True)
VOCAB_SIZE = len(tokenizer)

# 모델 설정
MODEL_CONFIGS = {
    "bilstm": {
        "class": BiLSTMAttention,
        "weight_path": "../Result/BiLSTM_v2/model/best_model.pth",
        "init_args": {
            "vocab_size": VOCAB_SIZE,
            "embed_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_classes": NUM_CLASSES
        }
    },
    "textcnn": {
        "class": TextCNN,
        "weight_path": "../Result/TextCNN/model/best_model.pth",
        "init_args": {
            "vocab_size": VOCAB_SIZE,
            "embed_dim": EMBEDDING_DIM,
            "num_classes": NUM_CLASSES
        }
    },
    "rcnn": {
        "class": RCNN,
        "weight_path": "../Result/RCNN/model/best_model.pth",
        "init_args": {
            "vocab_size": VOCAB_SIZE,
            "embed_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_classes": NUM_CLASSES
        }
    },
    "bert": {
        "class": KLUEBertModel,
        "weight_path": "../Result/kluebert_v1/model/best_model.pth",
        "init_args": {
            "bert_model_name": "klue/bert-base",
            "num_classes": NUM_CLASSES
        }
    }
}

# 파일 경로 설정
file_path = '../dataset/Interactive_Dataset/Interactive_VP_Dataset_kluebert_360_v1.csv'
precomputed_weights_path = '../token_weight/token_weights_kluebert.pt'
USE_TOKEN_WEIGHTS = True
MAX_LENGTH = 360


# 데이터셋 클래스 정의
class VoicePhishingDataset(Dataset):
    def __init__(self, use_precomputed_weights=True, precomputed_file=precomputed_weights_path):
        self.data = pd.read_csv(file_path, encoding='utf-8') if os.path.exists(file_path) else pd.read_csv(file_path,
                                                                                                           encoding='cp949')

        # 클래스 균형 맞추기 (보이스피싱 1, 정상 대화 0)
        phishing = self.data[self.data['label'] == 1]
        normal = self.data[self.data['label'] == 0].sample(n=len(phishing), random_state=42)
        self.data = pd.concat([phishing, normal]).sample(frac=1, random_state=42).reset_index(drop=True)

        # 토큰 가중치 불러오기
        self.precomputed_weights = torch.load(
            precomputed_file) if use_precomputed_weights and USE_TOKEN_WEIGHTS else None

        self.samples = []
        for i, (idx, row) in enumerate(self.data.iterrows()):
            text = str(row['transcript'])
            encoded = tokenizer(text, padding='max_length', truncation=True, max_length=MAX_LENGTH, return_tensors='pt')
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)

            # 사전 계산된 토큰 가중치 또는 1.0
            token_weights = self.precomputed_weights.get(idx, torch.ones_like(input_ids,
                                                                              dtype=torch.float)) if self.precomputed_weights else torch.ones_like(
                input_ids, dtype=torch.float)

            self.samples.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': torch.tensor(row['label'], dtype=torch.long),
                'token_weights': token_weights
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# 모델 로딩 함수
def load_model(model_class, weight_path, init_args):
    model = model_class(**init_args).to(DEVICE)
    state_dict = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# 사용할 모델 목록
model_keys = ["bilstm", "textcnn", "rcnn"]
if USE_BERT_MODEL:
    model_keys.append("bert")

models = [load_model(MODEL_CONFIGS[key]["class"], MODEL_CONFIGS[key]["weight_path"], MODEL_CONFIGS[key]["init_args"])
          for key in model_keys]


# 예측 함수 (attention_mask 포함)
def predict_model(model, input_tensor, attention_mask):
    with torch.no_grad():
        logits = model(input_tensor, attention_mask)
        probs = torch.softmax(logits, dim=1)
        return probs[0, 1].item()


# 메타 모델 정의
class MetaMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 8)
        self.relu = nn.GELU()
        self.fc2 = nn.Linear(8, num_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# 메타 피처 생성
def get_meta_features(dataset):
    meta_inputs, labels = [], []
    for sample in tqdm(dataset, desc="Generating meta features"):
        input_tensor = sample['input_ids'].unsqueeze(0).to(DEVICE)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(DEVICE)
        label = sample['label']

        # 각 모델에 대해 예측 결과를 받아옵니다.
        probs = [predict_model(model, input_tensor, attention_mask) for model in models]

        meta_inputs.append(probs)
        labels.append(label.item())
    return np.array(meta_inputs), np.array(labels)


# 메타 모델 학습 함수
def train_meta_model_torch(train_dataset):
    X_train, y_train = get_meta_features(train_dataset)
    X_train = torch.from_numpy(X_train).float().to(DEVICE)
    y_train = torch.from_numpy(y_train).long().to(DEVICE)

    model = MetaMLP(input_dim=X_train.shape[1], num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        preds = model(X_train)
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()

    # 메타 모델을 파일로 저장
    model_dir = "../Result/MetaMLP/models"
    os.makedirs(f"{model_dir}/Loss_plot", exist_ok=True)
    file_path = os.path.join(model_dir, "meta_model.pt")
    torch.save(model.state_dict(), file_path)

    return model  # 반환된 meta_model을 여기서 반환


# 메타 모델을 사용한 스태킹 예측
def predict_stacking_torch(input_tensor, attention_mask, meta_model, threshold=0.7):
    # 모델별 예측 확률을 가져옵니다.
    probs = [predict_model(model, input_tensor, attention_mask) for model in models]
    x = torch.tensor(probs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # meta_model을 사용해 예측을 수행합니다.
    meta_model.eval()
    with torch.no_grad():
        logits = meta_model(x)
        softmax_probs = torch.softmax(logits, dim=1).squeeze(0)  # [num_classes]

        # threshold 이상인 클래스가 있으면 그것으로 예측
        for i, p in enumerate(softmax_probs):
            if p.item() >= threshold:
                return i, p.item()

        # threshold를 넘지 않으면 argmax로 예측
        pred_class = torch.argmax(softmax_probs).item()
        return pred_class, softmax_probs[pred_class].item()


# Main
if __name__ == "__main__":
    print("[INFO] 데이터셋 로딩 중...")
    dataset = VoicePhishingDataset()
    print(f"[INFO] 학습 샘플 수: {len(dataset)}")

    print("[INFO] 메타 모델 학습 시작...")
    meta_model = train_meta_model_torch(dataset)  # 메타 모델 학습 및 반환

    print("[INFO] 예측 시작...")

    # 예시: input_tensor와 attention_mask 준비
    input_tensor = dataset[0]['input_ids'].unsqueeze(0).to(DEVICE)
    attention_mask = dataset[0]['attention_mask'].unsqueeze(0).to(DEVICE)

    # predict_stacking_torch 호출 시 meta_model 전달
    result_class, result_prob = predict_stacking_torch(input_tensor, attention_mask, meta_model)
    print(f"Predicted class: {result_class}, Probability: {result_prob}")

