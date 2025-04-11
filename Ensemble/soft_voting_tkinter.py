import tkinter as tk
from ensemble_soft_voting import predict_soft_voting
from matplotlib import font_manager
import matplotlib.pyplot as plt

font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows 경우
font_prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = font_prop.get_name()

class SoftVotingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("보이스피싱 탐지기 (Soft Voting)")
        self.root.geometry("600x600")
        self.threshold = 0.6
        self.warning_counter = 0
        self.warning_delay_active = False
        self.log_enabled = False
        self.debounce_after_id = None

        self.text_input = tk.Text(self.root, height=6, width=45, wrap='word', font=("Arial", 12))
        self.result_label = tk.Label(self.root, text="입력된 텍스트에 대한 결과가 여기에 표시됩니다.", font=("Arial", 12))
        self.threshold_label = tk.Label(self.root, text=f"Threshold: {self.threshold:.1f}", font=("Arial", 10))
        self.threshold_slider = tk.Scale(self.root, from_=0.0, to=1.0, resolution=0.1, orient="horizontal",
                                         command=self.update_threshold)
        self.threshold_entry = tk.Entry(self.root, width=6, justify="center", font=("Arial", 12))
        self.log_toggle_button = tk.Button(root, text="터미널 출력 OFF", bg='gray', font=("Arial", 10, "bold"),
                                           command=self.toggle_log_output)

        self.reset_button = tk.Button(root, text="경고 카운터 초기화", font=("Arial", 10), command=self.reset_warning)
        self.plot_button = tk.Button(root, text="확률 시각화", font=("Arial", 10), command=self.plot_probs)

        self.last_probs = None
        self.build_ui()

    def build_ui(self):
        tk.Label(self.root, text="메시지를 입력하세요:", font=("Arial", 12)).pack(pady=10)

        self.text_input.pack(pady=15)
        self.result_label.pack(pady=10)

        # Threshold 슬라이더
        self.threshold_label.pack(pady=5)
        self.threshold_slider.set(self.threshold)
        self.threshold_slider.pack(pady=10)

        # Threshold 입력
        tk.Label(self.root, text="Threshold 수동 입력 (0.00 ~ 1.00)").pack()
        self.threshold_entry.insert(0, str(self.threshold))
        self.threshold_entry.pack(pady=5)
        self.threshold_entry.bind('<Return>', self.entry_updated)

        # 로그 출력 토글
        self.log_toggle_button.config(command=self.toggle_log_output)
        self.log_toggle_button.pack(pady=5)

        self.reset_button.pack(pady=5)
        self.plot_button.pack(pady=5)

        self.text_input.bind('<KeyRelease>', self.debounced_prediction)

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

    def toggle_log_output(self):
        self.log_enabled = not self.log_enabled
        if self.log_enabled:
            self.log_toggle_button.config(text="터미널 출력 ON", bg='green')
        else:
            self.log_toggle_button.config(text="터미널 출력 OFF", bg='gray')

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

        prediction, confidence, probs = predict_soft_voting( text, self.threshold)
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

if __name__ == "__main__":
    root = tk.Tk()
    app = SoftVotingApp(root)
    root.mainloop()
