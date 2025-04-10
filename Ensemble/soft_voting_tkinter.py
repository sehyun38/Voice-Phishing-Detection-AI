import tkinter as tk
from ensemble_soft_voting import predict_soft_voting

class SoftVotingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("보이스피싱 탐지기 (Soft Voting)")
        self.root.geometry("500x450")  # 크기 약간 조정
        self.log_enabled = False
        self.debounce_after_id = None
        self.threshold = 0.6  # 기본 threshold 값을 0.6으로 설정
        self.build_ui()

    def build_ui(self):
        tk.Label(self.root, text="메시지를 입력하세요:", font=("Arial", 12)).pack(pady=10)

        self.text_input = tk.Text(self.root, height=6, width=45, wrap='word', font=("Arial", 12))
        self.text_input.pack(pady=15)

        self.result_label = tk.Label(self.root, text="입력된 텍스트에 대한 결과가 여기에 표시됩니다.", font=("Arial", 12))
        self.result_label.pack(pady=10)

        # Threshold 슬라이더
        self.threshold_label = tk.Label(self.root, text=f"Threshold: {self.threshold:.1f}", font=("Arial", 10))
        self.threshold_label.pack(pady=5)

        self.threshold_slider = tk.Scale(self.root, from_=0.0, to=1.0, resolution=0.1, orient="horizontal", command=self.update_threshold)
        self.threshold_slider.set(self.threshold)
        self.threshold_slider.pack(pady=10)

        # 로그 출력 토글
        self.log_toggle_button = tk.Button(self.root, text="터미널 출력 OFF", bg='gray', font=("Arial", 10, "bold"))
        self.log_toggle_button.config(command=self.toggle_log_output)
        self.log_toggle_button.pack(pady=5)

        self.text_input.bind('<KeyRelease>', self.debounced_prediction)

    def update_threshold(self, val):
        self.threshold = float(val)
        self.threshold_label.config(text=f"Threshold: {self.threshold:.2f}")  # Threshold 값을 UI에 업데이트

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

        prediction, confidence, probs = predict_soft_voting(text, threshold=self.threshold)

        if self.log_enabled:
            print("입력된 텍스트:", text)
            print("예측 결과:", "보이스피싱 의심" if prediction == 1 else "정상")
            print("확률 (정상 / 보이스피싱):", f"{probs[0] * 100:.2f}% / {probs[1] * 100:.2f}%\n")

        result_text = f"{'보이스피싱 의심' if prediction == 1 else '🟢 정상'} (확률: {confidence * 100:.2f}%)"
        result_color = 'red' if prediction == 1 else 'green'
        self.result_label.config(text=result_text, fg=result_color)

if __name__ == "__main__":
    root = tk.Tk()
    app = SoftVotingApp(root)
    root.mainloop()
