# 1. 라이브러리 및 기본 설정
import torch
import torch.nn as nn
import torch.nn.functional as F
import tkinter as tk
from transformers import AutoTokenizer, BertModel

embed_size = 256
num_classes = 2
max_length = 360
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = "CNN_v2"  # CNN, KLUEBERT
pooling = "mha"  # cls, mean, mha (KLUEBERT 전용)
USE_TOKEN_WEIGHTS = True
DEBUG_MODE = False

pretrained_model_name = "klue/bert-base"
model_path = f"Capaston/Result/{model_type}/model/best_model.pth"

AVAILABLE_MODELS = ["CNN_v2","CNN_v3", "KLUEBERT"]

# 2. 모델 정의
class CNNModel_v2(nn.Module):
    def __init__(self, vocab_size):
        super(CNNModel_v2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(embed_size, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Dropout(0.3),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
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
        return self.fc_layers(x)

class CNNModel_v3(nn.Module):
    def __init__(self, vocab_size):
        super(CNNModel_v3, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(embed_size, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Dropout(0.3),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
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
        return self.fc_layers(x)

# 3. MHA Pooling 모듈
class MHAPooling(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super(MHAPooling, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, sequence_output):
        batch_size = sequence_output.size(0)
        query = self.query.expand(batch_size, -1, -1)
        attn_output, _ = self.attention(query, sequence_output, sequence_output)
        return attn_output.squeeze(1)

# 4. KLUE BERT 모델
class KLUEBertModel(nn.Module):
    def __init__(self, bert_model_name="klue/bert-base", num_classes=2):
        super(KLUEBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size

        if pooling == 'mha':
            self.pooling = MHAPooling(hidden_size)
        elif pooling == 'mean':
            self.pooling = lambda x: torch.mean(x, dim=1)
        elif pooling == 'cls':
            self.pooling = lambda x: x[:, 0, :]
        else:
            raise ValueError("Invalid pooling type")

        self.fc1 = nn.Linear(hidden_size, 256)
        self.norm1 = nn.LayerNorm(256)
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(256, 128)
        self.norm2 = nn.LayerNorm(128)
        self.drop2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(128, 64)
        self.norm3 = nn.LayerNorm(64)
        self.drop3 = nn.Dropout(0.2)

        self.out = nn.Linear(64, num_classes)

    def forward(self, input_ids, attention_mask, token_weights=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        sequence_output = outputs.last_hidden_state

        if USE_TOKEN_WEIGHTS and token_weights is not None:
            token_weights = token_weights * attention_mask
            sequence_output = sequence_output * token_weights.unsqueeze(-1)

        pooled_output = self.pooling(sequence_output)

        x = self.fc1(pooled_output)
        x = nn.GELU()(x)
        x = self.norm1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = nn.GELU()(x)
        x = self.norm2(x)
        x = self.drop2(x)

        x = self.fc3(x)
        x = nn.GELU()(x)
        x = self.norm3(x)
        x = self.drop3(x)

        return self.out(x)

# 5. 모델 로딩
def load_model(model_path, vocab_size):
    if model_type == "CNN_v2":
        model = CNNModel_v2(vocab_size)
    elif model_type == "CNN_v3":
        model = CNNModel_v3(vocab_size)
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

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask, token_weights=attention_mask)

        probs = F.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0, prediction].item()

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
        self.build_ui()

    def build_ui(self):
        self.root.title("보이스피싱 탐지기")
        self.root.geometry("500x500")

        # 모델 선택
        model_frame = tk.Frame(self.root)
        model_frame.pack(pady=10)
        tk.Label(model_frame, text="모델 선택:", font=("Arial", 11)).pack(side="left")

        for m in AVAILABLE_MODELS:
            rb = tk.Radiobutton(model_frame, text=m, variable=self.model_var, value=m, command=self.on_model_change)
            rb.pack(side="left")

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

        model_path_dynamic = f"../Result/{model_type}/model/best_model.pth"
        self.model = load_model(model_path_dynamic, vocab_size)

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
