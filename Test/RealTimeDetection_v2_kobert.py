import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, BertModel
import tkinter as tk

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('monologg/kobert', trust_remote_code=True)

num_classes = 2
max_length = 360
keyword_weight_path = "../dataset/phishing_words.csv"
model_path = "../Result/model/kobert_v2/fold_2.pth"

# 키워드 가중치 불러오기
def load_keyword_weights(csv_path):
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except:
        df = pd.read_csv(csv_path, encoding="cp949")
    return {str(row['text']): float(row['weight']) for _, row in df.iterrows()}

# KoBERT 모델 정의
class KobertModel(nn.Module):
    def __init__(self, bert_model_name='monologg/kobert'):
        super(KobertModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)

        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(0.3),

            nn.Linear(64, num_classes)
        )

    def forward(self, input_ids, attention_mask, token_weights):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        sequence_output = outputs.last_hidden_state

        token_weights = token_weights.unsqueeze(-1)
        weighted_output = sequence_output * token_weights
        summed = torch.sum(weighted_output, dim=1)

        normed_weights = torch.clamp(token_weights.sum(dim=1), min=1e-3)
        pooled_output = summed / normed_weights

        logits = self.classifier(pooled_output)
        return logits

def load_model(path):
    model = KobertModel()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_token_weights(text, input_ids, keyword_weights):
    token_weights = torch.ones(len(input_ids[0]), dtype=torch.float)
    for i, token_id in enumerate(input_ids[0]):
        token = tokenizer.convert_ids_to_tokens(token_id.item())
        for keyword, weight in keyword_weights.items():
            if keyword in token.replace("▁", ""):
                token_weights[i] = weight
                break
    return token_weights.unsqueeze(0).to(device)

def predict(model, tokenizer, text, keyword_weights):
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    token_weights = get_token_weights(text, input_ids, keyword_weights)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, token_weights)
        probs = F.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0, prediction].item()
    return prediction, confidence, probs[0].cpu().numpy()

# GUI 클래스 정의
class VoicePhishingApp:
    def __init__(self, root, model, tokenizer, keyword_weights):
        self.root = root
        self.model = model
        self.tokenizer = tokenizer
        self.keyword_weights = keyword_weights
        self.log_enabled = False  # 기본값: 터미널 출력 OFF

        self.root.title("보이스피싱 실시간 탐지기")
        self.root.geometry("540x220")

        self.text_input = tk.Entry(root, width=70, font=("Arial", 14))
        self.text_input.pack(pady=15)

        self.result_label = tk.Label(root, text="텍스트 입력 후 예측 결과가 표시됩니다.", font=("Arial", 13))
        self.result_label.pack(pady=5)

        self.text_input.bind('<KeyRelease>', self.on_key_release)

        # 터미널 출력 토글 버튼
        self.toggle_button = tk.Button(root, text="터미널 출력 OFF", bg='gray',
                                       font=("Arial", 10, "bold"), command=self.toggle_logging)
        self.toggle_button.pack(pady=5)

    def on_key_release(self, event):
        text = self.text_input.get().strip()
        if not text:
            self.result_label.config(text="텍스트를 입력해주세요.", fg="black")
            return

        prediction, confidence, probs = predict(self.model, self.tokenizer, text, self.keyword_weights)

        if self.log_enabled:
            print("입력 텍스트:", text)
            print("예측 결과:", "보이스피싱 의심" if prediction == 1 else "정상")
            print("확률 (정상/보이스피싱): {:.2f}% / {:.2f}%\n".format(probs[0] * 100, probs[1] * 100))

        label_text = f"{'보이스피싱 의심' if prediction == 1 else '정상'} (확률: {confidence * 100:.2f}%)"
        self.result_label.config(text=label_text, fg="red" if prediction == 1 else "green")

    def toggle_logging(self):
        self.log_enabled = not self.log_enabled
        if self.log_enabled:
            self.toggle_button.config(text="터미널 출력 ON", bg="lightblue")
        else:
            self.toggle_button.config(text="터미널 출력 OFF", bg="gray")

# 실행 함수
def main():
    model = load_model(model_path)
    keyword_weights = load_keyword_weights(keyword_weight_path)

    root = tk.Tk()
    app = VoicePhishingApp(root, model, tokenizer, keyword_weights)
    root.mainloop()

if __name__ == "__main__":
    main()
