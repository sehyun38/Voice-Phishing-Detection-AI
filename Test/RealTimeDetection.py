import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('monologg/kobert', trust_remote_code=True)  # KoBertTokenizer로 변경

# 하이퍼파라미터 설정
num_classes = 2
max_length = 360

# 1모델 정의 (koBERT + 분류기)
class KobertModel(nn.Module):
    def __init__(self, bert_model_name='monologg/kobert'):
        super(KobertModel, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size  # 모델에 맞게 가져오기
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, 'pooler_output'):
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]  # [CLS] 토큰 벡터
        logits = self.classifier(pooled)
        return logits

# 2모델 불러오기
def load_model(model_path):
    model =  KobertModel()  # 모델 정의
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()
    return model

# 실시간 입력을 통한 예측 함수
def predict(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # 모델 예측
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=1)  # softmax로 확률값 계산
        prediction = torch.argmax(probs, dim=1).item()  # 가장 높은 확률의 클래스 예측
        confidence = probs[0, prediction].item()  # 예측된 클래스의 확률값

    return prediction, confidence

# 실시간 대화 및 예측
if __name__ == "__main__":
    model = load_model("../Result/model/kobert.pth")

    print("보이스피싱 여부를 실시간으로 확인합니다. 'exit'을 입력하면 종료됩니다.")

    while True:
        # 사용자로부터 입력 받기
        user_input = input("입력: ")

        # 'exit'을 입력하면 종료
        if user_input.lower() == 'exit':
            print("프로그램을 종료합니다.")
            break

        # 예측
        prediction, confidence = predict(model, tokenizer, user_input)

        # 예측 결과 출력
        if prediction == 1:
            print(f"보이스피싱 가능성 있음 (확률: {confidence * 100:.2f}%)")
        else:
            print(f"정상 (확률: {confidence * 100:.2f}%)")

