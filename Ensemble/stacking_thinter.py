import torch
import tkinter as tk
from transformers import AutoTokenizer

# 이제 import 가능
from ensemble_stacking import models, MetaMLP, predict_stacking_torch, DEVICE, NUM_CLASSES

# === 설정 ===
pretrained_model_name = "klue/bert-base"
max_length = 360

# === 메타 모델 로딩 ===
meta_model = MetaMLP(input_dim=len(models),num_classes=NUM_CLASSES).to(DEVICE)
meta_model.load_state_dict(torch.load("../Result/MetaMLP/models/meta_model.pt", map_location=DEVICE))
meta_model.eval()

# === 예측 함수 ===
def encode_text(tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
    return inputs['input_ids'].to(DEVICE).long()

def predict(text, tokenizer):
    input_tensor = encode_text(tokenizer, text)
    return predict_stacking_torch(input_tensor, meta_model)

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
            label, score = predict(text, self.tokenizer)
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
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    root = tk.Tk()
    app = VoicePhishingApp(root, tokenizer)
    root.mainloop()

if __name__ == "__main__":
    main()
