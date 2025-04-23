import tkinter as tk
from ensemble_weighted_voting import predict_weighted_voting, USE_BERT_MODEL
from matplotlib import font_manager
import matplotlib.pyplot as plt

# 한글 폰트
font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows 경우
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

# 모델 신뢰성 가중치
weights = [1.0, 1.0, 2.0]

if USE_BERT_MODEL:
    weights.append(0.1)

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

class SoftVotingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("보이스피싱 탐지기 (Soft Voting)")
        self.root.geometry("600x600")

        self.threshold = 0.6
        self.warning_manager = WarningManager()
        self.debounce_after_id = None
        self.log_enabled = False
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

        self.last_probs = None
        self.build_ui()

    def build_ui(self):
        tk.Label(self.root, text="메시지를 입력하세요:", font=("Arial", 12)).pack(pady=10)

        self.text_input.pack(pady=15)
        self.warning_label.pack(pady=5)

        self.result_label = tk.Label(self.root, text="입력된 텍스트에 대한 결과가 여기에 표시됩니다.", font=("Arial", 12))
        self.result_label.pack(pady=5)

        # Threshold 슬라이더
        self.threshold_slider.set(self.threshold)
        self.threshold_slider.pack(pady=10)

        # Threshold 입력
        tk.Label(self.root, text="Threshold 수동 입력 (0.00 ~ 1.00)").pack()
        self.threshold_entry.insert(0, str(self.threshold))
        self.threshold_entry.pack(pady=5)
        self.threshold_entry.bind('<Return>', self.entry_updated)

        # 로그 출력 토글
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

        prediction, confidence, probs = predict_weighted_voting( text, self.threshold, weights)
        self.last_probs = probs

        if self.warning_manager.should_warn(prediction):
            self.root.after(5000, self.warning_manager.reset_delay)

        # 메시지 분리
        result_msg, result_color = get_prediction_message(prediction, confidence)
        self.result_label.config(text=result_msg, fg=result_color)

        if self.warning_manager.counter > 0:
            warn_msg = f"누적 경고 {self.warning_manager.counter}회!"
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
        plt.bar(["일반 대화", "보이스피싱"], self.last_probs, color=["green", "red"])
        plt.ylim([0, 1])
        plt.title("Softmax 확률")
        plt.ylabel("확률")
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = SoftVotingApp(root)
    root.mainloop()
