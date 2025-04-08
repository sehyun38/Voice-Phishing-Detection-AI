import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import tkinter as tk


# 모델 로드 함수
def load_model(model_path, device):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model


# 예측 함수
def predict(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=360)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0, prediction].item()
    return prediction, confidence


# tkinter 연동
def update_text_input(event, text_input, model, tokenizer, device, result_label):
    text = text_input.get()
    prediction, confidence = predict(model, tokenizer, text, device)
    if prediction == 1:
        result_label.config(text=f"보이스피싱 의심 (확률: {confidence * 100:.2f}%)", fg="red")
    else:
        result_label.config(text=f"정상 (확률: {confidence * 100:.2f}%)", fg="green")


# GUI 실행
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

    # 모델 경로를 학습 때 저장한 경로로 지정 (예: fold3 모델)
    model = load_model("../Result/model/kobert_v1/fold4", device)

    root = tk.Tk()
    root.title("보이스피싱 탐지기")
    root.geometry("400x200")

    text_input = tk.Entry(root, width=60, font=("Arial", 14))
    text_input.pack(pady=20)

    result_label = tk.Label(root, text="입력된 텍스트에 대한 결과가 여기에 표시됩니다.", font=("Arial", 12))
    result_label.pack(pady=10)

    text_input.bind('<KeyRelease>',
                    lambda event: update_text_input(event, text_input, model, tokenizer, device, result_label))

    root.mainloop()


if __name__ == "__main__":
    main()
