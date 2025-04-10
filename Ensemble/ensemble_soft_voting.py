import torch
import tkinter as tk
from transformers import AutoTokenizer
from my_models import BiLSTMAttention, TextCNN, RCNN, KLUEBertModel

# === 설정 ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_BERT_MODEL = True  # True 시 BERT 모델 포함

EMBEDDING_DIM = 256
HIDDEN_DIM = 128
NUM_CLASSES = 2

# BERT tokenizer 기반 vocab size
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base", trust_remote_code=True)
VOCAB_SIZE = len(tokenizer)

# === 모델 설정 ===
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
    "bert": {
        "class": KLUEBertModel,
        "weight_path": "../Result/kluebert_v1/model/best_model.pth",
        "init_args": {
            "bert_model_name": "klue/bert-base",
            "num_classes": NUM_CLASSES
        }
    }
}


# 모델 로딩 함수
def load_model(model_class, weight_path, init_args):
    model = model_class(**init_args).to(DEVICE)
    state_dict = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# 모델 목록
model_keys = ["bilstm", "textcnn", "rcnn"]
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


# === 예측 함수 ===
def encode_text(tokenizer, text):
    """ 텍스트를 받아 input_ids와 attention_mask를 생성하여 반환 """
    encoded = tokenizer(
        text,
        padding="max_length",  # 최대 길이로 패딩
        truncation=True,  # 자르기
        max_length=360,  # 360으로 패딩 최대 길이 설정
        return_tensors="pt"
    )
    return encoded["input_ids"].to(DEVICE), encoded["attention_mask"].to(DEVICE)


def predict_model(model, input_ids, attention_mask=None):
    """ 모델에 대한 예측 함수 """
    with torch.no_grad():
        # 모델이 attention_mask를 사용하는지 체크하고 해당 입력을 전달
        if "attention_mask" in model.forward.__code__.co_varnames:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            logits = model(input_ids)
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy().tolist()
        return probs  # [p0, p1]


def predict_soft_voting(text, threshold=0.6):
    """ Soft Voting 방식 예측 """
    # 텍스트로부터 input_ids와 attention_mask 생성
    input_ids, attention_mask = encode_text(tokenizer, text)

    # 모든 모델에 대해 예측 확률 계산
    all_probs = [predict_model(model, input_ids, attention_mask) for model in models]  # [[p0, p1], ...]

    # 확률의 평균을 구함
    avg_probs = torch.tensor(all_probs).mean(dim=0).tolist()  # 평균 확률 [p0, p1]

    # threshold 적용 (class 1 확률이 threshold 이상이면 1, 아니면 0)
    prediction = int(avg_probs[1] >= threshold)  # class 1 확률 기준 판단
    confidence = avg_probs[1]

    return prediction, confidence, avg_probs


# === GUI 앱 ===
class VoicePhishingApp:
    def __init__(self, root, tokenizer):
        self.tokenizer = tokenizer
        self.root = root
        self.root.title("보이스피싱 탐지기 (Ensemble Stacking)")
        self.root.geometry("500x400")
        self.log_enabled = False
        self.debounce_after_id = None
        self.build_ui()

    def build_ui(self):
        tk.Label(self.root, text="메시지를 입력하세요:", font=("Arial", 12)).pack(pady=10)

        self.text_input = tk.Text(self.root, height=6, width=45, wrap='word', font=("Arial", 12))
        self.text_input.pack(pady=15)

        self.result_label = tk.Label(self.root, text="입력된 텍스트에 대한 결과가 여기에 표시됩니다.", font=("Arial", 12))
        self.result_label.pack(pady=10)

        # 로그 출력 토글
        self.log_toggle_button = tk.Button(self.root, text="터미널 출력 OFF", bg='gray', font=("Arial", 10, "bold"))
        self.log_toggle_button.config(command=self.toggle_log_output)
        self.log_toggle_button.pack(pady=5)

        self.text_input.bind('<KeyRelease>', self.debounced_prediction)

    def toggle_log_output(self):
        self.log_enabled = not self.log_enabled
        if self.log_enabled:
            self.log_toggle_button.config(text="터미널 출력 ON", bg='green')
        else:
            self.log_toggle_button.config(text="터미널 출력 OFF", bg='gray')

    def debounced_prediction(self, event, delay=300):
        if self.debounce_after_id is not None:
            self.root.after_cancel(self.debounce_after_id)
        self.debounce_after_id = self.root.after(delay, self.perform_prediction)

    def perform_prediction(self):
        text = self.text_input.get("1.0", "end").strip()
        if not text:
            self.result_label.config(text="텍스트를 입력해주세요.", fg="gray")
            return

        try:
            label, score, avg_probs = predict_soft_voting(text)  # 여기서 텍스트로 직접 예측 수행
            result_text = f"{'보이스피싱 의심' if label == 1 else '🟢 정상'} (확률: {score * 100:.2f}%)"
            result_color = 'red' if label == 1 else 'green'
            self.result_label.config(text=result_text, fg=result_color)

            if self.log_enabled:
                print("입력된 텍스트:", text)
                print("예측 결과:", "보이스피싱 의심" if label == 1 else "정상")
                print("확률:", f"{score * 100:.2f}%")
                print("-" * 40)

        except Exception as e:
            self.result_label.config(text=f"예측 중 오류 발생: {e}", fg="orange")
            if self.log_enabled:
                print("[오류]", str(e))


# === 메인 ===
def main():
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    root = tk.Tk()
    app = VoicePhishingApp(root, tokenizer)
    root.mainloop()


if __name__ == "__main__":
    main()
