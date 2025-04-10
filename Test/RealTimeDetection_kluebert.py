# 1. 라이브러리 및 기본 설정
import torch
import torch.nn as nn
import torch.nn.functional as F
import tkinter as tk
from transformers import AutoTokenizer, BertModel
from my_models import BiLSTMAttention, TextCNN, RCNN, KLUEBertModel, CNNModel_v2, CNNModel_v3

embed_size = 256
num_classes = 2
max_length = 360
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = "BiLSTM_v2"  # CNN, TextCNN, BiLSTM, RCNN, KLUEBERT
pooling = "mha"  # cls, mean, mha (KLUEBERT 전용)
USE_TOKEN_WEIGHTS = True
DEBUG_MODE = False

pretrained_model_name = "klue/bert-base"
model_path = f"../Result/{model_type}/model/best_model.pth"

AVAILABLE_MODELS = ["CNN_v2", "CNN_v3", "TextCNN", "BiLSTM","BiLSTM_v2", "RCNN", "KLUEBERT"]

# 5. 모델 로딩
def load_model(model_path, vocab_size):
    print(f"[DEBUG] 모델 타입: {model_type}")  # 디버깅 메시지 추가
    if model_type == "CNN_v2":
        model = CNNModel_v2(vocab_size)
    elif model_type == "CNN_v3":
        model = CNNModel_v3(vocab_size)
    elif model_type == "TextCNN":
        model = TextCNN(vocab_size)
    elif model_type == "BiLSTM":
        model = BiLSTMAttention(vocab_size)
    elif model_type == "BiLSTM_v2":
        model = BiLSTMAttention(vocab_size)
    elif model_type == "RCNN":
        model = RCNN(vocab_size)
    elif model_type == "KLUEBERT":
        model = KLUEBertModel(pretrained_model_name, num_classes)
    else:
        raise ValueError("지원하지 않는 모델 타입입니다")

    model = model.to(device)

    if model_type != "KLUEBERT":
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    return model

# 6. 예측 함수
def predict(model, tokenizer, text):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=max_length
    )
    input_ids = inputs['input_ids'].to(device).long()
    attention_mask = inputs['attention_mask'].to(device).float()

    THRESHOLD = 0.6

    with torch.no_grad():
        if model_type == "KLUEBERT":
            logits = model(input_ids, attention_mask=attention_mask, token_weights=attention_mask)
        else:
            logits = model(input_ids, token_weights=attention_mask)

        probs = F.softmax(logits, dim=1)
        class1_prob = probs[0, 1].item()  # 보이스피싱 확률
        if class1_prob > THRESHOLD:
            prediction = 1  # 보이스피싱
        else:
            prediction = 0  # 일반 대화

        confidence = class1_prob  # 확신도는 클래스 1에 대한 것 기준

    return prediction, confidence, probs[0].cpu().numpy()

# 7. GUI 클래스
class VoicePhishingApp:
    def __init__(self, root, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.root = root
        self.log_enabled = False
        self.debounce_after_id = None
        self.model_var = tk.StringVar(value=model_type)
        self.threshold = 0.6  # 초기 THRESHOLD 값 설정
        self.build_ui()

    # 모델 선택을 위한 체크박스를 2줄로 배치
    def build_ui(self):
        self.root.title("보이스피싱 탐지기")
        self.root.geometry("500x500")  # 크기 확장

        # 모델 선택
        model_frame = tk.Frame(self.root)
        model_frame.pack(pady=10)
        tk.Label(model_frame, text="모델 선택:", font=("Arial", 11)).grid(row=0, column=0, columnspan=2)  # Label을 grid로 배치

        # 2줄로 체크박스를 배치하기 위한 설정
        row = 1  # 두 번째 행부터 시작
        for idx, m in enumerate(AVAILABLE_MODELS):
            rb = tk.Radiobutton(model_frame, text=m, variable=self.model_var, value=m, command=self.on_model_change)
            rb.grid(row=row, column=idx % 5, padx=5, pady=5, sticky="w")

            if (idx + 1) % 5 == 0:  # 5개 항목 후 줄 바꿈
                row += 1

        # 텍스트 입력
        self.text_input = tk.Text(self.root, height=6, width=45, wrap='word', font=("Arial", 12))
        self.text_input.pack(pady=15)

        # 결과 출력
        self.result_label = tk.Label(self.root, text="입력된 텍스트에 대한 결과가 여기에 표시됩니다.", font=("Arial", 12))
        self.result_label.pack(pady=10)

        # 로그 출력 토글
        self.log_toggle_button = tk.Button(self.root, text="터미널 출력 OFF", bg='gray', font=("Arial", 10, "bold"))
        self.log_toggle_button.config(command=self.toggle_log_output)
        self.log_toggle_button.pack(pady=5)

        # 슬라이더 추가: THRESHOLD 값 설정
        self.threshold_slider = tk.Scale(self.root, from_=0, to=1, resolution=0.01, orient="horizontal",
                                         label="Threshold", command=self.update_threshold)
        self.threshold_slider.set(self.threshold)  # 초기값 설정
        self.threshold_slider.pack(pady=10)

        # 입력 이벤트
        self.text_input.bind('<KeyRelease>', self.debounced_prediction)

    def on_model_change(self):
        selected_model = self.model_var.get()
        self.change_model(selected_model)

    def change_model(self, selected_model_type):
        global model_type
        model_type = selected_model_type

        if self.log_enabled:
            print(f"[INFO] 선택된 모델: {model_type}")

        vocab_size = len(self.tokenizer)

        # 동적으로 모델 경로 설정
        model_path_dynamic = f"../Result/{model_type}/model/best_model.pth"
        self.model = load_model(model_path_dynamic, vocab_size)

    def toggle_log_output(self):
        self.log_enabled = not self.log_enabled
        if self.log_enabled:
            self.log_toggle_button.config(text="터미널 출력 ON", bg='green')
        else:
            self.log_toggle_button.config(text="터미널 출력 OFF", bg='gray')

    def update_threshold(self, val):
        # 슬라이더 값이 변경되면 threshold 업데이트
        self.threshold = float(val)

    def debounced_prediction(self, event, delay=100):
        if self.debounce_after_id is not None:
            self.root.after_cancel(self.debounce_after_id)
        self.debounce_after_id = self.root.after(delay, self.perform_prediction)

    def perform_prediction(self):
        text = self.text_input.get("1.0", "end").strip()
        if not text:
            self.result_label.config(text="텍스트를 입력해주세요.")
            return

        prediction, confidence, probs = predict(self.model, self.tokenizer, text)

        if self.log_enabled:
            print("입력된 텍스트:", text)
            print("예측 결과:", "보이스피싱 의심" if prediction == 1 else "정상")
            print("확률 (정상 / 보이스피싱):", f"{probs[0] * 100:.2f}% / {probs[1] * 100:.2f}%\n")

        # 슬라이더 값에 따른 결과 표시
        result_text = f"{'보이스피싱 의심' if prediction == 1 else '정상'} (확률: {confidence * 100:.2f}%)"
        result_color = 'red' if prediction == 1 else 'green'
        self.result_label.config(text=result_text, fg=result_color)

# 8. 메인 함수
def main():
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)
    vocab_size = len(tokenizer)
    model = load_model(model_path, vocab_size)

    root = tk.Tk()
    app = VoicePhishingApp(root, model, tokenizer)
    root.mainloop()

if __name__ == "__main__":
    main()
