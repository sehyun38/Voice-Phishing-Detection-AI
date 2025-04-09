# ensemble_soft_voting.py
import torch
from transformers import AutoTokenizer
from my_models import BiLSTMAttention, TextCNN, RCNN, CNNModel_v3, KLUEBertModel

# === 설정 ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_BERT_MODEL = False  # True 시 BERT 모델 포함

EMBEDDING_DIM = 256
HIDDEN_DIM = 128
NUM_CLASSES = 2

tokenizer = AutoTokenizer.from_pretrained("klue/bert-base", trust_remote_code=True)

# === MODEL_CONFIGS ===
VOCAB_SIZE = len(tokenizer)
MODEL_CONFIGS = {
    "bilstm": {
        "class": BiLSTMAttention,
        "weight_path": "../Result/BiLSTM/model/best_model.pth",
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
    "cnn_v3": {
        "class": CNNModel_v3,
        "weight_path": "../Result/CNN_v3/model/best_model.pth",
        "init_args": {
            "vocab_size": VOCAB_SIZE
        }
    },
    "bert": {
        "class": KLUEBertModel,
        "weight_path": "../Result/Klue_Bert/model/best_model.pth",
        "init_args": {
            "bert_model_name": "klue/bert-base",
            "num_classes": NUM_CLASSES
        }
    }
}

def load_model(model_class, weight_path, init_args):
    model = model_class(**init_args).to(DEVICE)
    state_dict = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model_keys = ["bilstm", "textcnn", "rcnn", "cnn_v3"]
if USE_BERT_MODEL:
    model_keys.append("bert")

models = [
    load_model(
        MODEL_CONFIGS[key]["class"],
        MODEL_CONFIGS[key]["weight_path"],
        MODEL_CONFIGS[key]["init_args"]
    )
    for key in model_keys
]

def prepare_input(text):
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    return encoded["input_ids"].to(DEVICE), encoded["attention_mask"].to(DEVICE)

def predict_model(model, input_ids, attention_mask=None):
    with torch.no_grad():
        if "attention_mask" in model.forward.__code__.co_varnames:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            logits = model(input_ids)
        if logits.shape[-1] == 1:
            prob = torch.sigmoid(logits).item()
            return [1 - prob, prob]
        else:
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy().tolist()
            return probs

def predict_soft_voting(text):
    input_ids, attention_mask = prepare_input(text)
    all_probs = [predict_model(model, input_ids, attention_mask) for model in models]
    avg_probs = torch.tensor(all_probs).mean(dim=0).tolist()
    prediction = int(avg_probs[1] > 0.5)
    confidence = avg_probs[1]
    return prediction, confidence, avg_probs
