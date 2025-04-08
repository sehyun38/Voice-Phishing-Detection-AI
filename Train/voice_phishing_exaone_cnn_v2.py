"""
Tokenizer : EXAONE 3.5 최대 32,768개의 토큰까지 가능(4096 설정)
Model : CNN
optimizer : Adam
epochs : 100
batch size : 32
조기종료 :  5회
csv : 1(보이스 피싱), 0(일상 대화) 비율 1:1 불용어 제거, 중요 키워드 가중치 계산 , 인코딩 utf-8(cp949)
cross-validation 사용, ROCAUC
"""

import os, sys, torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
import shutil

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# 하이퍼파라미터 설정
batch_size = 32
num_epochs = 100
embed_size = 256
num_classes = 2
max_length = 360

# 파일 경로 설정
file_path = '../dataset/Interactive_Dataset/Interactive_VP_Dataset_exaone_360_v3.csv'
token_weights_dict = torch.load('../token_weight/token_weights_exaone.pt', weights_only=True)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct", trust_remote_code=True)

class VoicePhishingDataset(Dataset):
    def __init__(self, verbose=False):
        try:
            self.data = pd.read_csv(file_path, encoding='utf-8')
        except:
            self.data = pd.read_csv(file_path, encoding='cp949')

        phishing = self.data[self.data['label'] == 1]
        normal = self.data[self.data['label'] == 0].sample(n=len(phishing), random_state=42)

        self.data = pd.concat([phishing, normal]).sample(frac=1, random_state=42).reset_index(drop=True)

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

            # token_weights 길이 조정
            raw_weights = token_weights_dict[idx]
            if len(raw_weights) < max_length:
                padded_weights = torch.cat([raw_weights, torch.zeros(max_length - len(raw_weights))])
            else:
                padded_weights = raw_weights[:max_length]

            sample = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_weights': padded_weights,
                'label': torch.tensor(row['label'], dtype=torch.long)
            }

            if verbose:
                visualize_token_weights(sample, tokenizer, verbose=True)

            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class CNNModel(nn.Module):
    def __init__(self, vocab_size):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.cnn_layers = nn.Sequential(
            nn.Conv1d(embed_size, 512, kernel_size=3, padding=1),  # 채널 수 증가
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
            nn.Flatten(),  # (batch, 256)
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask=None, token_weights=None):
        x = self.embedding(input_ids)  # (batch, seq_len, embed_size)

        if token_weights is not None and attention_mask is not None:
            token_weights = token_weights * attention_mask  # 패딩 부분 제거

        if token_weights is not None:
            x = x * token_weights.unsqueeze(-1)  # 패딩 부분을 처리

        x = x.permute(0, 2, 1)  # (batch, embed_size, seq_len)
        x = self.cnn_layers(x)  # (batch, 256, 1)
        x = x.flatten(start_dim=1)  # (batch, 256)
        x = self.fc_layers(x)  # (batch, num_classes)
        return x

#토큰과 해당 가중치를 시각화하는 함수.
def visualize_token_weights(sample, tokenizer, verbose=False):
    if not verbose:
        return

    input_ids = sample['input_ids']
    weights = sample['token_weights']

    # 토큰 변환
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    weight_values = weights.tolist()

    print(f"{'Index':<5} {'Token':<20} {'Weight':<6}")
    print("-" * 35)
    for i, (tok, w) in enumerate(zip(tokens, weight_values)):
        print(f"{i:<5} {tok:<20} {w:<6.4f}")

# ROC curve plotting 함수
def plot_roc_curve(fold, all_labels, all_probs, save_dir='../Result/ROCAUC/CNN_v2', MODE = "SAVE"):
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc = roc_auc_score(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f'Fold {fold} (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Fold {fold}')
    plt.legend(loc='lower right')

    #ROC 저장
    if MODE == "SAVE":
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'roc_curve_fold_{fold}.png'))

    # ROC 보기
    elif MODE == "SHOW":
        plt.show()

    plt.close()

# 배치별 평가 결과 집계 함수
def accumulate_predictions(model, dataloader):
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

# 메트릭 계산 함수
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

    plot_roc_curve(fold, metrics['all_labels'], metrics['all_probs'], MODE ="SAVE")

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

        case_dir = f'../Result/case_samples/CNN_v2/fold_{fold}'
        os.makedirs(case_dir, exist_ok=True)

        for case_num in [1, 2, 3, 4]:
            case_df = val_data[val_data['case'] == case_num]
            case_path = os.path.join(case_dir, f'case_{case_num}.csv')
            case_df.to_csv(case_path, index=False, encoding='utf-8-sig')
            print(f"Case {case_num} 샘플 {len(case_df)}개 저장 완료: {case_path}")

# 최종 평가 함수
def evaluate(model, dataloader):
    total_loss, all_preds, all_labels, all_probs = accumulate_predictions(model, dataloader)
    return compute_metrics(total_loss, all_preds, all_labels, all_probs)

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

def train(model, dataloader, optimizer, scaler, fold, epoch, epochs):
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

        sys.stdout.write(f"\rFold {fold} | Epoch {epoch}/{epochs} | Batch {batch_idx}/{total_batches} | Loss: {loss.item():.5f}")
        sys.stdout.flush()

    print()
    avg_loss = total_loss / len(dataloader)
    train_acc = accuracy_score(all_labels, all_preds)
    print(f"Fold {fold} | Epoch {epoch} Completed | Loss: {avg_loss:.5f} | Acc: {train_acc:.4f}\n")
    return avg_loss

if __name__ == '__main__':
    dataset = VoicePhishingDataset(verbose=False)
    labels = dataset.data['label'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = {'loss': [], 'acc': [], 'roc_auc': [], 'true_acc': [], 'false_acc': []}

    best_roc_auc = 0
    best_fold = None
    best_model_file = None

    fold = 1
    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        print(f"--- Fold {fold} ---")
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size)

        vocab_size = len(tokenizer)
        model = CNNModel(vocab_size).to(device)
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr= 2e-5, weight_decay=5e-5)
        early_stopping = EarlyStopping(patience=30, min_delta=0.001)

        scaler = GradScaler()

        for epoch in range(1, num_epochs + 1):
            train_loss = train(model, train_loader, optimizer, scaler, fold, epoch, num_epochs)
            metrics = evaluate(model, val_loader)
            if early_stopping(metrics['avg_loss']):
                report_evaluation(fold, metrics)
                break

        fold_metrics['loss'].append(metrics['avg_loss'])
        fold_metrics['acc'].append(metrics['accuracy'])
        fold_metrics['roc_auc'].append(metrics['roc_auc'])
        fold_metrics['true_acc'].append(metrics['true_acc'])
        fold_metrics['false_acc'].append(metrics['false_acc'])

        model_dir = '../Result/model/CNN_v2'
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