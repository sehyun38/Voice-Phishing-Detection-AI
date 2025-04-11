import torch

from Train.voice_phishing_kluebert_v1 import pooling
from config import  VOCAB_SIZE, DEVICE,tokenizer, MAX_LENGTH
from models import  BiLSTMAttention, TextCNN, RCNN, KLUEBertModel

# True 시 BERT 모델 포함
USE_BERT_MODEL = True

# 하이퍼 파라미터
batch_size = 32
num_epochs = 20
num_classes = 2
embed_size = 256
hidden_dim = 128
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
        # 모델이 attention_mask 를 사용하는지 체크하고 해당 입력을 전달
        if "attention_mask" in model.forward.__code__.co_varnames:
            logits = model(input_ids, attention_mask)
        else:
            logits = model(input_ids)
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy().tolist()
        return probs


def predict_soft_voting(text, threshold=0.6):
    """ Soft Voting 방식 예측 """
    # 텍스트로부터 input_ids와 attention_mask 생성
    input_ids, attention_mask = encode_text(text)

    # 모든 모델에 대해 예측 확률 계산
    models = load_all_models()
    all_probs = [predict_model(model, input_ids, attention_mask) for model in models]  # [[p0, p1], ...]

    # 확률의 평균을 구함
    avg_probs = torch.tensor(all_probs).mean(dim=0).tolist()  # 평균 확률 [p0, p1]

    # threshold 적용 (class 1 확률이 threshold 이상이면 1, 아니면 0)
    prediction = int(avg_probs[1] >= threshold)  # class 1 확률 기준 판단
    confidence = avg_probs[1]

    return prediction, confidence, avg_probs