import torch
import os
from concurrent.futures import ThreadPoolExecutor

from Train.voice_phishing_kluebert_v1 import pooling
from config import  VOCAB_SIZE, DEVICE,tokenizer, MAX_LENGTH, num_classes
from models import  BiLSTMAttention, TextCNN, RCNN, KLUEBertModel

# 하이퍼 파라미터
embed_size = 256
hidden_dim = 128

# 앙상블 모델 BERT fine-tuning 모델 추가
USE_BERT_MODEL = True
#모델 구조와 가중치 파일 로드
model_structure_load = True

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
        weight_path = f"../Result/{folder_names[key]}/model"

        if key == "bilstm":
            configs[key] = {
                "class": BiLSTMAttention,
                "weight_path":os.path.join(weight_path,"best_model.pth"),
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
                "weight_path":os.path.join(weight_path,"best_model.pth"),
                "init_args": {
                    "vocab_size": VOCAB_SIZE,
                    "embed_dim": embed_size,
                    "num_classes": num_classes
                }
            }
        elif key == "rcnn":
            configs[key] = {
                "class": RCNN,
                "weight_path": os.path.join(weight_path,"best_model.pth"),
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
                "weight_path": os.path.join(weight_path,"best_model.pth"),
                "init_args": {
                    "pooling": pooling,
                    "num_classes": num_classes
                }
            }

    return configs

# 3. TorchScript 경로 (.pt)
MODEL_PATHS_PT = {
    "bilstm": {
        "class": BiLSTMAttention,
        "weight_path": os.path.join("../Result", "BiLSTM_v2", "model","best_model.pt")
    },
    "textcnn": {
        "class": TextCNN,
        "weight_path": os.path.join("../Result", "TextCNN", "model", "best_model.pt")
    },
    "rcnn": {
        "class": RCNN,
        "weight_path": os.path.join("../Result", "RCNN", "model", "best_model.pt")
    },
    "bert": {
        "class": KLUEBertModel,
        "weight_path": os.path.join("../Result", "kluebert_v1", "model", "fold_1.pt")
    }
}

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

    # 전역에서 만든 MODEL_CONFIGS 를 사용
    for key in model_keys:
        if model_structure_load:
            model = load_scripted_model(MODEL_PATHS_PT[key]["weight_path"])
        else:
            config = model_configs[key]
            model = load_model_weights(
                config["class"],
                config["weight_path"],
                config["init_args"]
            )
        models.append(model)
    return models

# === 예측 함수 ===
def encode_text(text):
    encoded = tokenizer(
        text,
        padding="max_length",  # 최대 길이로 패딩
        truncation=True,  # 자르기
        max_length= MAX_LENGTH,  # 360으로 패딩 최대 길이 설정
        return_tensors="pt"
    )
    return encoded["input_ids"].to(DEVICE), encoded["attention_mask"].to(DEVICE)


def predict_model(model, input_ids, attention_mask=None):
    """ 모델에 대한 예측 함수 """
    with torch.no_grad():
        for key in model_keys:
            if key == "TextCNN" or key =="RCNN":
                logits = model(input_ids)
            else:
                logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy().tolist()
        return probs


def predict_weighted_voting(text, threshold=0.6, weights=None):
    """ Weighted Soft Voting 방식 예측 """
    # 텍스트로부터 input_ids와 attention_mask 생성
    input_ids, attention_mask = encode_text(text)

    # 모델 불러오기
    models = load_all_models()  # 3개 모델 [RCNN, TextCNN, BiLSTM]

    # 기본 가중치 설정 (동일 가중치면 Soft Voting과 동일)
    if weights is None:
        weights = [1.0] * len(models)

    # 각 모델의 확률 예측
    with ThreadPoolExecutor() as executor:
        all_probs = list(executor.map(
            lambda model: predict_model(model, input_ids, attention_mask),
            models
        ))
    # all_probs = [predict_model(model, input_ids , attention_mask) for model in models]  # [[p0, p1], ...]

    # Weighted 평균 계산
    weighted_sum = torch.tensor([0.0, 0.0])
    total_weight = sum(weights)
    for prob, w in zip(all_probs, weights):
        weighted_sum += torch.tensor(prob) * w
    avg_probs = (weighted_sum / total_weight).tolist()  # 가중 평균 확률 [p0, p1]

    # threshold 기반 클래스 결정
    prediction = int(avg_probs[1] >= threshold)
    confidence = avg_probs[1]

    return prediction, confidence, avg_probs
