import os
import torch
import torch.nn.functional as f
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib import font_manager

from config import DEVICE, tokenizer, VOCAB_SIZE, MAX_LENGTH, num_classes
from models import BiLSTMAttention, TextCNN, RCNN, KLUEBertModel, CNNModelV2, CNNModelV3

font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows 경우
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

# 하이퍼 파라미터
embed_size = 256
hidden_size = 128

model_type = "BiLSTM_v2"
pooling = "mha"
model_structure_load = True  # True -> .pt, False -> .pth
file_ext = ".pt" if model_structure_load else ".pth"

AVAILABLE_MODELS = ["CNN_v2", "CNN_v3", "TextCNN", "BiLSTM_v2", "RCNN", "KLUEBERT"]
AVAILABLE_FOLDS = ["best_model", "fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {model_path}")

    if model_type != "KLUEBERT":
        if model_structure_load:
            model = torch.jit.load(model_path, map_location=DEVICE)
        else:
            if model_type == "CNN_v2":
                model = CNNModelV2(VOCAB_SIZE, embed_size, num_classes)
            elif model_type == "CNN_v3":
                model = CNNModelV3(VOCAB_SIZE, embed_size, num_classes)
            elif model_type == "TextCNN":
                model = TextCNN(VOCAB_SIZE, embed_size, num_classes)
            elif model_type == "BiLSTM_v2":
                model = BiLSTMAttention(VOCAB_SIZE, embed_size, hidden_size, num_classes)
            elif model_type == "RCNN":
                model = RCNN(VOCAB_SIZE, embed_size, hidden_size, num_classes)
            elif model_type =="KLUEBERT":
                model = KLUEBertModel(pooling, num_classes)
            else:
                raise ValueError("지원하지 않는 모델 타입입니다")

            model = model.to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        return model

def predict(model, text, threshold=0.6):
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LENGTH)
    input_ids = inputs['input_ids'].to(DEVICE).long()
    attention_mask = inputs['attention_mask'].to(DEVICE).float()

    with torch.no_grad():
        if model_type == "TextCNN" or model_type == "RCNN":
            logits = model(input_ids)
        else:
            logits = model(input_ids, attention_mask)

        probs = f.softmax(logits, dim=1)
        class1_prob = probs[0, 1].item()
        prediction = int(class1_prob > threshold)
        return prediction, class1_prob, probs[0].cpu().numpy()

class VoicePhishingApp:
    def __init__(self, root, model):
        self.model = model
        self.root = root
        self.root.title("보이스피싱 탐지기")
        self.root.geometry("600x700")
        self.threshold = 0.6
        self.warning_counter = 0
        self.warning_delay_active = False  # 🔥 딜레이 플래그
        self.log_enabled = False
        self.debounce_after_id = None
        self.model_var = tk.StringVar(value=model_type)
        self.fold_var = tk.StringVar(value="best_model")

        self.text_input = tk.Text(root, height=6, width=45, wrap='word', font=("Arial", 12))
        self.result_label = tk.Label(root, text="입력된 텍스트에 대한 결과가 여기에 표시됩니다.", font=("Arial", 12))
        self.threshold_slider = tk.Scale(self.root, from_=0.0, to=1.0, resolution=0.1, orient="horizontal",
                                         label="Threshold", command=self.slider_updated)
        self.threshold_entry = tk.Entry(self.root, width=6, justify="center", font=("Arial", 12))
        self.log_toggle_button = tk.Button(root, text="터미널 출력 OFF", bg='gray', font=("Arial", 10, "bold"),
                                           command=self.toggle_log_output)
        self.reset_button = tk.Button(root, text="경고 카운터 초기화", font=("Arial", 10), command=self.reset_warning)
        self.plot_button = tk.Button(root, text="확률 시각화", font=("Arial", 10), command=self.plot_probs)

        self.last_probs = None
        self.build_ui()

    def build_ui(self):
        # 모델 선택
        model_frame = tk.Frame(self.root)
        model_frame.pack(pady=10)
        tk.Label(model_frame, text="모델 선택:", font=("Arial", 11)).grid(row=0, column=0, sticky="w")
        for i, m in enumerate(AVAILABLE_MODELS):
            rb = tk.Radiobutton(model_frame, text=m, variable=self.model_var, value=m, command=self.on_model_change)
            rb.grid(row=(i // 5) + 1, column=i % 5, padx=5, pady=5, sticky="w")

        # 폴드 선택
        fold_frame = tk.Frame(self.root)
        fold_frame.pack(pady=10)
        tk.Label(fold_frame, text="폴드 선택:", font=("Arial", 11)).grid(row=0, column=0, sticky="w")
        for i, fold in enumerate(AVAILABLE_FOLDS):
            rb = tk.Radiobutton(fold_frame, text=fold, variable=self.fold_var, value=fold, command=self.on_model_change)
            rb.grid(row=(i // 6) + 1, column=i % 6, padx=5, pady=5, sticky="w")

        self.text_input.pack(pady=15)
        self.result_label.pack(pady=10)

        self.threshold_slider.set(self.threshold)
        self.threshold_slider.pack(pady=10)

        tk.Label(self.root, text="Threshold 수동 입력 (0.00 ~ 1.00)").pack()
        self.threshold_entry.insert(0, str(self.threshold))
        self.threshold_entry.pack(pady=5)
        self.threshold_entry.bind('<Return>', self.entry_updated)

        self.log_toggle_button.pack(pady=5)

        self.reset_button.pack(pady=5)
        self.plot_button.pack(pady=5)

        self.text_input.bind('<KeyRelease>', self.debounced_prediction)

    def on_model_change(self):
        global model_type
        model_type = self.model_var.get()
        fold_name = self.fold_var.get()

        model_filename = "best_model" if fold_name == "best_model" else fold_name
        model_path = f"../Result/{model_type}/model/{model_filename}{file_ext}"

        if not os.path.exists(model_path):
            self.result_label.config(text=f"모델 파일 없음: {model_path}", fg="red")
            return

        try:
            self.model = load_model(model_path)
            self.warning_counter = 0
            self.warning_delay_active = False  # 🔥 딜레이도 리셋
            self.result_label.config(text=f"{model_type}/{fold_name} 모델 로드 완료 ({file_ext})", fg="blue")
            self.perform_prediction()

        except Exception as e:
            self.result_label.config(text=f"모델 로드 실패: {e}", fg="red")

    def toggle_log_output(self):
        self.log_enabled = not self.log_enabled
        text = "터미널 출력 ON" if self.log_enabled else "터미널 출력 OFF"
        bg = "green" if self.log_enabled else "gray"
        self.log_toggle_button.config(text=text, bg=bg)

    def update_threshold(self, val):
        self.threshold = float(val)

    def slider_updated(self, val):
        val = round(float(val), 2)
        self.threshold = val
        self.threshold_entry.delete(0, tk.END)
        self.threshold_entry.insert(0, str(val))

    def entry_updated(self, event=None):
        try:
            val = float(self.threshold_entry.get())
            if 0.0 <= val <= 1.0:
                self.threshold = val
                self.threshold_slider.set(val)
            else:
                raise ValueError
        except ValueError:
            self.result_label.config(text="0.00~1.00 사이 실수를 입력하세요", fg="orange")

    def reset_warning(self):
        self.warning_counter = 0
        self.warning_delay_active = False  # 🔥 딜레이도 리셋
        self.result_label.config(text="경고 카운터가 초기화되었습니다", fg="blue")

    def debounced_prediction(self, event, delay=100):
        if self.debounce_after_id:
            self.root.after_cancel(self.debounce_after_id)
        self.debounce_after_id = self.root.after(delay, self.perform_prediction)

    def perform_prediction(self):
        text = self.text_input.get("1.0", "end").strip()

        try:
            self.threshold = float(self.threshold_entry.get())
            if not (0.0 <= self.threshold <= 1.0):
                raise ValueError
        except ValueError:
            self.result_label.config(text="Threshold는 0.00 ~ 1.00 사이 실수여야 해요!", fg="orange")
            return

        if not text:
            self.result_label.config(text="텍스트를 입력해주세요.", fg="orange")
            return

        prediction, confidence, probs = predict(self.model, text, self.threshold)
        self.last_probs = probs

        if prediction == 1:
            if not self.warning_delay_active:
                self.warning_counter += 1
                self.warning_delay_active = True
                self.root.after(5000, self.reset_warning_delay)  # 🔥 10초 후 경고 다시 가능

            if self.warning_counter >= 3:
                result = f"누적 경고 {self.warning_counter}회! 보이스피싱 의심 (확률: {confidence * 100:.2f}%)"
                color = "red"
            else:
                result = f"경고 {self.warning_counter}/3 (확률: {confidence * 100:.2f}%)"
                color = "orange"
        else:
            result = f"정상 (확률: {confidence * 100:.2f}%)"
            color = "green"

        self.result_label.config(text=result, fg=color)

        if self.log_enabled:
            print(f"[입력] {text}")
            print(f"[확률] {probs}")
            print(f"[결과] {result}\n")

    def reset_warning_delay(self):
        self.warning_delay_active = False

    def plot_probs(self):
        if self.last_probs is None:
            self.result_label.config(text="먼저 예측을 수행하세요.")
            return
        plt.clf()
        plt.bar(["일반 대화", "보이스피싱"], self.last_probs, color=["green", "red"])
        plt.ylim([0, 1])
        plt.title("Softmax 확률")
        plt.ylabel("확률")
        plt.show()


def main():
    model_filename = "best_model" + file_ext
    model_path = f"../Result/{model_type}/model/{model_filename}"
    model = load_model(model_path)
    root = tk.Tk()
    app = VoicePhishingApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
