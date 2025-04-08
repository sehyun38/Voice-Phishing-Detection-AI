"""
Tokenizer : koBert (4096 설정)
Model : Bertr기반 koBert Classifier
optimizer : AdamW
epochs : 20
batch size : 32
조기종료 :  5회
csv : 1(보이스 피싱), 0(일상 대화) 비율 1:1 불용어 제거, 중요 키워드 가중치 계산 , 인코딩 utf-8(cp949)
cross-validation 사용, ROCAUC, GradScaler, cosine-Schedule
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BertModel, AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from torch.amp import GradScaler, autocast
import shutil

# 디바이스 설정 (CUDA 사용 가능 시 GPU 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# 하이퍼파라미터 설정
batch_size = 32
num_epochs = 20
num_classes = 2
max_length = 360

# 데이터 파일 경로
file_path = '../dataset/Interactive_Dataset/Interactive_VP_Dataset_kobert_360_v3.csv'
# 미리 계산된 토큰 가중치 파일 경로
precomputed_weights_path = '../token_weight/token_weights_kobert.pt'

# KoBERT 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained('monologg/kobert', trust_remote_code=True)

# 학습용 데이터셋 클래스 (미리 계산된 토큰 가중치 사용 옵션 추가)
class VoicePhishingDataset(Dataset):
    def __init__(self, use_precomputed_weights=False, precomputed_file=precomputed_weights_path):
        try:
            self.data = pd.read_csv(file_path, encoding='utf-8')
        except:
            self.data = pd.read_csv(file_path, encoding='cp949')

        # 피싱과 정상 데이터 균형 맞추기
        phishing = self.data[self.data['label'] == 1]
        normal = self.data[self.data['label'] == 0].sample(n=len(phishing), random_state=42)

        self.data = pd.concat([phishing, normal]).sample(frac=1, random_state=42).reset_index(drop=True)
        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)

        # 미리 계산된 토큰 가중치 불러오기 (데이터셋 생성 순서와 동일해야 함)
        if use_precomputed_weights:
            self.precomputed_weights = torch.load(precomputed_file, weights_only=True)
        else:
            self.precomputed_weights = None

        self.samples = []
        for idx, row in self.data.iterrows():
            text = str(row['transcript'])
            encoded = tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
            # 미리 계산된 토큰 가중치 사용 (없으면 기본값 1.0)
            if self.precomputed_weights is not None:
                token_weights = self.precomputed_weights[idx]
            else:
                token_weights = torch.ones_like(input_ids, dtype=torch.float)
            self.samples.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': torch.tensor(row['label'], dtype=torch.long),
                'token_weights': token_weights
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class MHAPooling(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super(MHAPooling, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))  # 학습 가능한 query

    def forward(self, sequence_output):
        batch_size = sequence_output.size(0)
        query = self.query.expand(batch_size, -1, -1)  # (batch_size, 1, hidden)
        attn_output, _ = self.attention(query, sequence_output, sequence_output)
        return attn_output.squeeze(1)  # (batch_size, hidden)

# 모델 클래스 정의
class KobertModel(nn.Module):
    def __init__(self, bert_model_name='monologg/kobert', num_classes=2):
        super(KobertModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)

        self.mha_pooling = MHAPooling(hidden_size=self.bert.config.hidden_size, num_heads=8)

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
        sequence_output = sequence_output * token_weights

        pooled_output = self.mha_pooling(sequence_output)
        logits = self.classifier(pooled_output)
        return logits

# ROC 곡선 그리기 함수
def plot_roc_curve(fold, all_labels, all_probs, save_dir='../Result/ROCAUC/kobert_v3',MODE = "SAVE"):
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc = roc_auc_score(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f'Fold {fold} (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Fold {fold}')
    plt.legend(loc='lower right')

    # ROC 저장
    if MODE == "SAVE":
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'roc_curve_fold_{fold}.png'))

    # ROC 보기
    elif MODE == "SHOW":
        plt.show()

    plt.close()

# 모델 예측 누적 함수
def accumulate_predictions(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            token_weights = batch['token_weights'].to(device)

            logits = model(input_ids, attention_mask, token_weights)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return total_loss, all_preds, all_labels, all_probs

# 평가 메트릭 계산 함수
def compute_metrics(total_loss, all_preds, all_labels, all_probs):
    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    true_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_acc = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'avg_loss': avg_loss,
        'accuracy': acc,
        'roc_auc': roc_auc,
        'true_acc': true_acc,
        'false_acc': false_acc,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probs': all_probs
    }

# 평가 결과 출력 + 케이스별 저장
def report_evaluation(fold, metrics, dataset=None, val_idx=None):
    print(f"Fold {fold} Evaluation Report:")
    print(classification_report(metrics['all_labels'], metrics['all_preds'], digits=4))
    print(f"Accuracy: {metrics['accuracy']:.4f}, ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"True Positive Rate: {metrics['true_acc']:.4f}")
    print(f"True Negative Rate: {metrics['false_acc']:.4f}\n")

    plot_roc_curve(fold, metrics['all_labels'], metrics['all_probs'])

    # 케이스 라벨링 및 저장
    if dataset is not None and val_idx is not None:
        true_labels = np.array(metrics['all_labels'])
        pred_labels = np.array(metrics['all_preds'])
        new_labels = np.zeros_like(true_labels)

        new_labels[(true_labels == 1) & (pred_labels == 1)] = 1  # TP
        new_labels[(true_labels == 0) & (pred_labels == 1)] = 2  # FP
        new_labels[(true_labels == 1) & (pred_labels == 0)] = 3  # FN
        new_labels[(true_labels == 0) & (pred_labels == 0)] = 4  # TN

        val_data = dataset.data.iloc[val_idx].copy()
        val_data['prediction'] = pred_labels
        val_data['case'] = new_labels

        case_dir = f'../Result/case_samples/kobert_v3/fold_{fold}'
        os.makedirs(case_dir, exist_ok=True)

        for case_word in ['TP', 'FP', 'FN', 'TN']:
            case_df = val_data[val_data['case'] == case_word]
            case_path = os.path.join(case_dir, f'case_{case_word}.csv')
            case_df.to_csv(case_path, index=False, encoding='utf-8-sig')
            print(f"Case {case_word} 샘플 {len(case_df)}개 저장 완료: {case_path}")

# 평가 함수
def evaluate(model, dataloader, criterion):
    total_loss, all_preds, all_labels, all_probs = accumulate_predictions(model, dataloader, criterion)
    return compute_metrics(total_loss, all_preds, all_labels, all_probs)

# 조기 종료 클래스
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

# 학습 함수
def train(model, dataloader, optimizer, scaler, criterion, fold, epoch, epochs):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    total_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader, start=1):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        token_weights = batch['token_weights'].to(device)

        optimizer.zero_grad()

        if device == "cuda":
            with autocast(device_type='cuda'):
                logits = model(input_ids, attention_mask, token_weights)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

        else:
            logits = model(input_ids, attention_mask, token_weights)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        sys.stdout.write(f"\rFold {fold} | Epoch {epoch}/{epochs} | Batch {batch_idx}/{total_batches} | Loss: {loss.item():.7f}")
        sys.stdout.flush()

    print()
    avg_loss = total_loss / len(dataloader)
    train_acc = accuracy_score(all_labels, all_preds)
    print(f"Fold {fold} | Epoch {epoch} Completed | Loss: {avg_loss:.5f} | Acc: {train_acc:.4f}\n")
    return avg_loss

if __name__ == "__main__":
    # use_precomputed_weights 옵션 True로 설정하여 미리 계산된 토큰 가중치 사용
    dataset = VoicePhishingDataset(use_precomputed_weights=True)
    labels = dataset.data['label'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = {'loss': [], 'acc': [], 'roc_auc': [], 'true_acc': [], 'false_acc': []}

    best_roc_auc = 0
    best_fold = None
    best_model_file = None
    fold = 1

    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        print(f"\n===== Fold {fold} =====")
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size)

        model = KobertModel().to(device)
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(model.parameters(), lr=4e-6 , weight_decay=1e-5)
        early_stopping = EarlyStopping(patience=5, min_delta=0.001)

        scaler = GradScaler()

        for epoch in range(1, num_epochs + 1):
            train_loss = train(model, train_loader, optimizer, scaler, criterion, fold, epoch, num_epochs)
            metrics = evaluate(model, val_loader, criterion)

            if early_stopping(metrics['avg_loss']):
                report_evaluation(fold, metrics)    #dataset, val_idx
                break

        fold_metrics['loss'].append(metrics['avg_loss'])
        fold_metrics['acc'].append(metrics['accuracy'])
        fold_metrics['roc_auc'].append(metrics['roc_auc'])
        fold_metrics['true_acc'].append(metrics['true_acc'])
        fold_metrics['false_acc'].append(metrics['false_acc'])

        model_dir = '../Result/model/kobert_v3'
        os.makedirs(model_dir, exist_ok=True)
        fold_model_file = os.path.join(model_dir, f'fold_{fold}.pth')
        torch.save(model.state_dict(), fold_model_file)
        print(f"Fold {fold} 모델 저장 완료: {fold_model_file}\n")

        if metrics['roc_auc'] > best_roc_auc:
            best_roc_auc = metrics['roc_auc']
            best_fold = fold
            best_model_file = fold_model_file

        fold += 1

    print("=== Overall Metrics ===")
    for key, values in fold_metrics.items():
        print(f"Average {key.capitalize()}: {np.mean(values):.4f}")

    print(f"\n최적의 모델은 Fold {best_fold}이며, 모델 파일 이름은 '{best_model_file}' 입니다.")
    final_model_name = os.path.join(model_dir, 'best_model.pth')
    shutil.copy(best_model_file, final_model_name)