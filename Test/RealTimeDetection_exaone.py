# 1. 라이브러리 및 설정
import torch
import torch.nn as nn
import torch.nn.functional as F
import tkinter as tk
from transformers import AutoTokenizer

embed_size = 256
num_classes = 2
max_length = 360
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = "BiLSTM"  # CNN, TextCNN, BiLSTM, RCNN
model_path = f"../Result/model/{model_type.lower()}/best_model.pth" #fold3.pth
pretrained_model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

# 2. 모델 정의
class CNNModel(nn.Module):
    def __init__(self, vocab_size):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(embed_size, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(0.3)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask=None, token_weights=None):
        x = self.embedding(input_ids)
        if token_weights is not None and attention_mask is not None:
            token_weights = token_weights * attention_mask
        if token_weights is not None:
            x = x * token_weights.unsqueeze(-1)
        x = x.permute(0, 2, 1)
        x = self.cnn_layers(x)
        x = x.flatten(start_dim=1)
        x = self.fc_layers(x)
        return x

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_classes=2, kernel_sizes=[3, 4, 5], num_channels=100):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_channels, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_channels * len(kernel_sizes), num_classes)

    def forward(self, input_ids, attention_mask=None, token_weights=None):
        x = self.embedding(input_ids)
        if token_weights is not None:
            x = x * token_weights.unsqueeze(-1)
        x = x.permute(0, 2, 1)
        x = [torch.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return self.fc(x)

class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=128, num_classes=2):
        super(BiLSTMAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention_fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask=None, token_weights=None):
        x = self.embedding(input_ids)
        if token_weights is not None:
            x = x * token_weights.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attention_fc(lstm_out).squeeze(-1)
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        if token_weights is not None:
            attn_scores = attn_scores * token_weights
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = (lstm_out * attn_weights).sum(dim=1)
        context = self.dropout(context)
        return self.fc(context)

class RCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=128, num_classes=2):
        super(RCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(in_channels=hidden_dim * 2, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask=None, token_weights=None):
        x = self.embedding(input_ids)
        if token_weights is not None:
            x = x * token_weights.unsqueeze(-1)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = torch.max(x, dim=2)[0]
        x = self.dropout(x)
        return self.fc(x)

# 3. 모델 로딩 함수
def load_model(model_path, vocab_size):
    if model_type == "CNN":
        model = CNNModel(vocab_size)
    elif model_type == "TextCNN":
        model = TextCNN(vocab_size)
    elif model_type == "BiLSTM":
        model = BiLSTMAttention(vocab_size)
    elif model_type == "RCNN":
        model = RCNN(vocab_size)
    else:
        raise ValueError(f"지원하지 않는 모델 타입입니다: {model_type}")

    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 4. 예측 함수
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

    with torch.no_grad():
        logits = model(input_ids, token_weights=attention_mask)
        probs = F.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0, prediction].item()

    return prediction, confidence, probs[0].cpu().numpy()

# 5. GUI 애플리케이션 클래스
class VoicePhishingApp:
    def __init__(self, root, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.root = root
        self.log_enabled = False
        self.debounce_after_id = None
        self.build_ui()

    def build_ui(self):
        self.root.title("보이스피싱 탐지기")
        self.root.geometry("450x300")

        self.text_input = tk.Text(self.root, height=6, width=45, wrap='word', font=("Arial", 12))
        self.text_input.pack(pady=15)

        self.result_label = tk.Label(self.root, text="입력된 텍스트에 대한 결과가 여기에 표시됩니다.", font=("Arial", 12))
        self.result_label.pack(pady=10)

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

        result_text = f"{'보이스피싱 의심' if prediction == 1 else '정상'} (확률: {confidence * 100:.2f}%)"
        result_color = 'red' if prediction == 1 else 'green'
        self.result_label.config(text=result_text, fg=result_color)

# 6. 메인 함수
def main():
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)
    vocab_size = len(tokenizer)
    model = load_model(model_path, vocab_size)

    root = tk.Tk()
    app = VoicePhishingApp(root, model, tokenizer)
    root.mainloop()

if __name__ == "__main__":
    main()
