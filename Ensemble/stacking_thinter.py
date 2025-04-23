import os
import torch
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib import font_manager

from ensemble_stacking import load_all_models, MetaMLP, predict_stacking_torch, model_structure_load
from config import DEVICE, tokenizer, MAX_LENGTH, num_classes

# 한글 폰트
font_path = "C:/Windows/Fonts/malgun.ttf"
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

# 모델 경로
model_dir = "../Result/MetaMLP/models"

# === 메타 모델 로딩 ===
if model_structure_load:
    META_PATH = os.path.join(model_dir, "meta_model.pt")
    meta_model = torch.jit.load(META_PATH, map_location=DEVICE)
else:
    META_PATH = os.path.join(model_dir, "meta_model.pth")
    models = load_all_models()
    meta_model = MetaMLP(input_dim=len(models)).to(DEVICE)
    meta_model.load_state_dict(torch.load(META_PATH, map_location=DEVICE))
    meta_model.eval()

# === 텍스트 인코딩 및 예측 ===
def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LENGTH)
    return inputs['input_ids'].to(DEVICE).long(), inputs['attention_mask'].to(DEVICE)

def predict(text, threshold):
    input_tensor, attention_mask = encode_text(text)
    return predict_stacking_torch(input_tensor, attention_mask, meta_model, threshold)

# === 신뢰도 기반 메시지 생성 ===
def get_prediction_message(prediction, confidence):
    confidence_percent = confidence * 100
    if prediction == 1:
        if confidence >= 0.85:
            return f"매우 높은 확률로 보이스피싱입니다 ({confidence_percent:.2f}%)", "red"
        elif confidence >= 0.65:
            return f"보이스피싱 가능성 있음 ({confidence_percent:.2f}%)", "orange"
        else:
            return f"의심되지만 확신은 부족 ({confidence_percent:.2f}%)", "orange"
    else:
        return f"정상으로 판단됨 ({confidence_percent:.2f}%)", "green"

# === 경고 상태 관리 클래스 ===
class WarningManager:
    def __init__(self):
        self.counter = 0
        self.delay_active = False

    def should_warn(self, prediction):
        if prediction == 1 and not self.delay_active:
            self.counter += 1
            self.delay_active = True
            return True
        return False

    def reset_delay(self):
        self.delay_active = False

    def reset(self):
        self.counter = 0
        self.delay_active = False

# === GUI 앱 ===
class VoicePhishingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("보이스피싱 탐지기 (Ensemble Stacking)")
        self.root.geometry("600x650")

        self.threshold = 0.5
        self.warning_manager = WarningManager()
        self.log_enabled = False
        self.debounce_after_id = None
        self.last_probs = None

        self.text_input = tk.Text(self.root, height=6, width=45, wrap='word', font=("Arial", 12))
        self.warning_label = tk.Label(self.root, text="", font=("Arial", 12))
        self.result_label = tk.Label(self.root, text="입력된 텍스트에 대한 결과가 여기에 표시됩니다.", font=("Arial", 12))
        self.threshold_slider = tk.Scale(self.root, from_=0.0, to=1.0, resolution=0.1,
                                         orient="horizontal", label="Threshold", command=self.slider_updated)
        self.threshold_entry = tk.Entry(self.root, width=6, justify="center", font=("Arial", 12))
        self.log_toggle_button = tk.Button(self.root, text="터미널 출력 OFF", bg='gray',
                                           font=("Arial", 10, "bold"), command=self.toggle_log_output)
        self.reset_button = tk.Button(self.root, text="경고 카운터 초기화", font=("Arial", 10), command=self.reset_warning)
        self.plot_button = tk.Button(self.root, text="확률 시각화", font=("Arial", 10), command=self.plot_probs)

        self.build_ui()

    def build_ui(self):
        tk.Label(self.root, text="메시지를 입력하세요:", font=("Arial", 12)).pack(pady=10)

        self.text_input.pack(pady=15)
        self.warning_label.pack(pady=5)

        self.result_label = tk.Label(self.root, text="입력된 텍스트에 대한 결과가 여기에 표시됩니다.", font=("Arial", 12))
        self.result_label.pack(pady=5)

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

    def toggle_log_output(self):
        self.log_enabled = not self.log_enabled
        if self.log_enabled:
            self.log_toggle_button.config(text="터미널 출력 ON", bg='green')
        else:
            self.log_toggle_button.config(text="터미널 출력 OFF", bg='gray')

    def slider_updated(self, val):
        val = round(float(val), 2)
        self.threshold = val
        self.threshold_entry.delete(0, tk.END)
        self.threshold_entry.insert(0, str(val))
        self.reset_warning()

    def entry_updated(self, event=None):
        try:
            val = float(self.threshold_entry.get())
            if 0.0 <= val <= 1.0:
                self.threshold = val
                self.threshold_slider.set(val)
                self.reset_warning()
            else:
                raise ValueError
        except ValueError:
            self.result_label.config(text="0.00~1.00 사이 실수를 입력하세요", fg="orange")

    def reset_warning(self):
        self.warning_manager.reset()
        self.warning_label.config(text="경고 카운터가 초기화되었습니다", fg="blue")

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

        prediction, confidence, probs = predict(text, self.threshold)
        self.last_probs = probs

        if self.warning_manager.should_warn(prediction):
            self.root.after(5000, self.warning_manager.reset_delay)

        # 메시지 분리
        result_msg, result_color = get_prediction_message(prediction, confidence)
        self.result_label.config(text=result_msg, fg=result_color)

        if self.warning_manager.counter > 0:
            warn_msg = f"❗ 누적 경고 {self.warning_manager.counter}회!"
            self.warning_label.config(text=warn_msg, fg="orange" if self.warning_manager.counter < 3 else "red")
        else:
            self.warning_label.config(text="")

        if self.log_enabled:
            print(f"[입력] {text}")
            print(f"[확률] {probs}")
            print(f"[결과] {result_msg} / 경고: {self.warning_manager.counter}회\n")

    def plot_probs(self):
        if self.last_probs is None:
            self.result_label.config(text="먼저 예측을 수행하세요.")
            return
        plt.clf()
        class_names = ["일반 대화", "보이스피싱"] if num_classes == 2 else [f"클래스 {i}" for i in range(num_classes)]
        plt.bar(class_names, self.last_probs, color=["green", "red"][:num_classes])
        plt.ylim([0, 1])
        plt.title("Softmax 확률")
        plt.ylabel("확률")
        plt.show()

# === 실행 ===
def main():
    root = tk.Tk()
    app = VoicePhishingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
