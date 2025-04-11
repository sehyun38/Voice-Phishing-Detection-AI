import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from Train.voice_phishing_kluebert_v1 import pooling
from config import VOCAB_SIZE, DEVICE, FILE_PATH, PT_SAVE_PATH, tokenizer, MAX_LENGTH
from models import BiLSTMAttention, TextCNN, RCNN, KLUEBertModel
from utils import VoicePhishingDataset

# BERT 모델 포함 여부
USE_BERT_MODEL = True
USE_TOKEN_WEIGHTS = True

#하이퍼 파라미터
embed_size = 256
hidden_dim = 128
num_classes = 2

# 모델 목록
model_keys = ["bilstm", "textcnn", "rcnn"]
if USE_BERT_MODEL:
    model_keys.append("bert")

# 2. 실제 파일 폴더명 정의
MODEL_FOLDER_NAMES = {
    "bilstm": "BiLSTM_v2",
    "textcnn": "TextCNN",
    "rcnn": "RCNN",
    "bert": "kluebert_v1"
}

# === 모델 설정 ===
def generate_model_configs(folder_names):
    configs = {}

    for key in model_keys:
        weight_path = f"../Result/{folder_names[key]}/model/best_model.pth"

        if key == "bilstm":
            configs[key] = {
                "class": BiLSTMAttention,
                "weight_path": weight_path,
                "init_args": {
                    "vocab_size": VOCAB_SIZE,
                    "embed_dim": embed_size,
                    "hidden_dim": hidden_dim,
                    "num_classes": num_classes
                }
            }
        elif key == "textcnn":
            configs[key] = {
                "class": TextCNN,
                "weight_path": weight_path,
                "init_args": {
                    "vocab_size": VOCAB_SIZE,
                    "embed_dim": embed_size,
                    "num_classes": num_classes
                }
            }
        elif key == "rcnn":
            configs[key] = {
                "class": RCNN,
                "weight_path": weight_path,
                "init_args": {
                    "vocab_size": VOCAB_SIZE,
                    "embed_dim": embed_size,
                    "hidden_dim": hidden_dim,
                    "num_classes": num_classes
                }
            }
        elif key == "bert":
            configs[key] = {
                "class": KLUEBertModel,
                "weight_path": weight_path,
                "init_args": {
                    "pooling": pooling,
                    "num_classes": num_classes
                }
            }

    return configs

# 3. TorchScript 경로 (.pt)
MODEL_PATHS_PT = {
    key: f"../Result/{MODEL_FOLDER_NAMES[key]}/model/best_model.pt"
    for key in model_keys
}

model_structure_load = True

# 모델 로딩 함수
def load_model_weights(model_class, weight_path, init_args):
    model = model_class(**init_args).to(DEVICE)
    state_dict = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_scripted_model(weight_path):
    model = torch.jit.load(weight_path, map_location=DEVICE)
    model.eval()
    return model

def load_all_models():
    models = []
    model_configs = generate_model_configs(MODEL_FOLDER_NAMES)

    # 전역에서 만든 MODEL_CONFIGS를 사용
    for key in model_keys:
        if model_structure_load:
            model = load_scripted_model(MODEL_PATHS_PT[key])
        else:
            config = model_configs[key]
            model = load_model_weights(
                config["class"],
                config["weight_path"],
                config["init_args"]
            )
        models.append(model)
    return models

# 예측 함수 (attention_mask 포함)
def predict_model(model, input_tensor, attention_mask):
    with torch.no_grad():
        logits = model(input_tensor, attention_mask)
        probs = torch.softmax(logits, dim=1)
        return probs[0, 1].item()

# 메타 모델 정의
class MetaMLP(nn.Module):
    def __init__(self, input_dim):
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
        models = load_all_models()
        probs = [predict_model(model, input_tensor, attention_mask) for model in models]

        meta_inputs.append(probs)
        labels.append(label.item())
    return np.array(meta_inputs), np.array(labels)


# 메타 모델 학습 함수
def train_meta_model_torch(train_dataset):
    X_train, y_train = get_meta_features(train_dataset)
    X_train = torch.from_numpy(X_train).float().to(DEVICE)
    y_train = torch.from_numpy(y_train).long().to(DEVICE)

    model = MetaMLP(input_dim=X_train.shape[1]).to(DEVICE)
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
    if model_structure_load:
        model_path = os.path.join(model_dir, "meta_model.pt")
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, model_path)
    else:
        model_path = os.path.join(model_dir, "meta_model.pth")
        torch.save(model.state_dict(), model_path)

    return model  # 반환된 meta_model을 여기서 반환


# 메타 모델을 사용한 스태킹 예측
def predict_stacking_torch(input_tensor, attention_mask, meta_model, threshold=0.6):
    # 모델별 예측 확률을 가져옵니다.
    models = load_all_models()
    probs = [predict_model(model, input_tensor, attention_mask) for model in models]
    x = torch.tensor(probs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # meta_model을 사용해 예측을 수행합니다.
    meta_model.eval()
    with torch.no_grad():
        logits = meta_model(x)
        softmax_probs = torch.softmax(logits, dim=1).squeeze(0)  # [num_classes]

        # threshold 이상인 클래스가 있으면 그것으로 예측
        for i, p in enumerate(softmax_probs):
            p: torch.Tensor  # <- 명시적으로 타입 힌트 추가
            if p.item() >= threshold:
                return i, p.item()

        # threshold를 넘지 않으면 argmax로 예측
        pred_class = torch.argmax(softmax_probs).item()
        return pred_class, softmax_probs[pred_class].item()

# Main
if __name__ == "__main__":
    print("[INFO] 데이터셋 로딩 중...")
    dataset = VoicePhishingDataset(FILE_PATH, PT_SAVE_PATH, tokenizer, MAX_LENGTH, USE_TOKEN_WEIGHTS)
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

