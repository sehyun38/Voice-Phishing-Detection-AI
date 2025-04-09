"""
Tokenizer : KlueBert 최대 512(360 설정)
Model : BiLSTM
optimizer : Adam
epochs : 20
batch size : 32
조기종료 :  30회
csv : 1(보이스 피싱), 0(일상 대화) 비율 1:1 불용어 제거, 중요 키워드 가중치 계산 , 인코딩 utf-8(cp949)
cross-validation 사용, ROCAUC, 학습 스케줄러(warmup_cosine_annealing), GradScaler
"""

import os, sys
import torch
import math
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

# KlueBert 토큰화 모델 로드
tokenizer = AutoTokenizer.from_pretrained(	"klue/bert-base", trust_remote_code=True)

# 하이퍼파라미터 설정
batch_size = 32
num_epochs = 100
embed_size = 256
num_classes = 2

# 파일 경로 설정
file_path =  '../dataset/Interactive_Dataset/Interactive_VP_Dataset_kluebert_360_v1.csv'
precomputed_weights_path = '../token_weight/token_weights_kluebert.pt'

#그래프 저장 "SAVE", "SHOW"
MODE = "SAVE"

#토큰, 가중치 출력
DEBUG_MODE = False

#토큰 가중치 제어
USE_TOKEN_WEIGHTS = True
MAX_LENGTH = 360

# 데이터셋 클래스 정의 (각 단어 가중치 추가)
class VoicePhishingDataset(Dataset):
    def __init__(self, use_precomputed_weights=True, precomputed_file=precomputed_weights_path):
        try:
            self.data = pd.read_csv(file_path, encoding='utf-8')
        except:
            self.data = pd.read_csv(file_path, encoding='cp949')

        # 클래스 균형 맞추기
        phishing = self.data[self.data['label'] == 1]
        normal = self.data[self.data['label'] == 0].sample(n=len(phishing), random_state=42)
        self.data = pd.concat([phishing, normal]).sample(frac=1, random_state=42).reset_index(drop=True)

        # 토큰 가중치 불러오기
        self.precomputed_weights = torch.load(precomputed_file) if use_precomputed_weights and USE_TOKEN_WEIGHTS else None

        self.samples = []
        for i, (idx, row) in enumerate(self.data.iterrows()):
            text = str(row['transcript'])
            encoded = tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)

            # 사전 계산된 가중치 또는 1.0
            if self.precomputed_weights is not None:
                token_weights = self.precomputed_weights.get(idx, torch.ones_like(input_ids, dtype=torch.float))
            else:
                token_weights = torch.ones_like(input_ids, dtype=torch.float)

            self.samples.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': torch.tensor(row['label'], dtype=torch.long),
                'token_weights': token_weights
            })

            # 선택적 디버깅 출력
            MAX_TOKENS_TO_PRINT = 10  # 출력할 최대 토큰 수

            if DEBUG_MODE and i < 3:
                print(f"\n[DEBUG] Sample Index: {idx}")
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                weights = token_weights.tolist()
                for i, (tok, w) in enumerate(zip(tokens, weights)):
                    if tok not in ['[PAD]']:
                        if i >= MAX_TOKENS_TO_PRINT:
                            break
                        print(f"{i:03} | {repr(tok):<15} | weight: {w:.4f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# BiLSTM 모델 정의
class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=128, num_classes=2):
        super(BiLSTMAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention_fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask=None, token_weights=None):
        x = self.embedding(input_ids)  # (B, L, D)
        if token_weights is not None:
            x = x * token_weights.unsqueeze(-1)

        lstm_out, _ = self.lstm(x)  # (B, L, 2H)
        attn_scores = self.attention_fc(lstm_out).squeeze(-1)  # (B, L)

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        if token_weights is not None:
            attn_scores = attn_scores * token_weights  # 보정

        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, L, 1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (B, 2H)

        context = self.dropout(context)
        return self.fc(context)

def warmup_cosine_annealing(epoch):
    warmup_epochs = 5
    total_epochs = num_epochs
    min_lr_factor = 0.1

    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr_factor + 0.5 * (1 - min_lr_factor) * (1 + math.cos(progress * math.pi))

#loss, accuracy 그래프 함수
def plot_metrics_from_lists(fold,train_metrics, val_metrics, save_path=f'../Result/BiLSTM'):
    epochs = range(1, len(train_metrics) + 1)

    train_loss = [m['avg_loss'] for m in train_metrics]
    val_loss = [m['avg_loss'] for m in val_metrics]
    train_acc = [m['accuracy'] for m in train_metrics]
    val_acc = [m['accuracy'] for m in val_metrics]

    # Loss
    plt.figure()
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    os.makedirs(f"{save_path}/Loss_plot", exist_ok=True)
    if MODE == "SAVE":
        plt.savefig(os.path.join(save_path, f'Loss_plot/loss_curve_fold_{fold}.png'))
    elif MODE == "SHOW":
        plt.show()
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    os.makedirs(f"{save_path}/Accuracy_Plot", exist_ok=True)
    if MODE == "SAVE":
        plt.savefig(os.path.join(save_path, f'Accuracy_Plot/accuracy_curve_fold_{fold}.png'))
    elif MODE == "SHOW":
        plt.show()
    plt.close()

# ROC curve plotting 함수
def plot_roc_curve(fold, all_labels, all_probs, save_dir='../Result/BiLSTM/ROCAUC'):
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
            THRESHOLD = 0.7
            preds = (probs > THRESHOLD).long()

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

# 평가 결과를 텍스트로 출력하는 함수
def print_evaluation_summary(fold, metrics):
    print(f"\nFold {fold} Evaluation Report:")
    print(classification_report(metrics['all_labels'], metrics['all_preds'], digits=4))
    print(f"Accuracy: {metrics['accuracy']:.4f}, ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"True Positive Rate: {metrics['true_acc']:.4f}")
    print(f"True Negative Rate: {metrics['false_acc']:.4f}\n")

# 예측값과 정답값을 비교하여 케이스(TP, FP, FN, TN) 라벨을 생성하는 함수
def classify_cases(true_labels, pred_labels):
    new_labels = np.zeros_like(true_labels)
    new_labels[(true_labels == 1) & (pred_labels == 1)] = 1  # TP
    new_labels[(true_labels == 0) & (pred_labels == 1)] = 2  # FP
    new_labels[(true_labels == 1) & (pred_labels == 0)] = 3  # FN
    new_labels[(true_labels == 0) & (pred_labels == 0)] = 4  # TN
    return new_labels

# 케이스별 샘플을 저장하고, 전부 비어 있을 경우 전체 데이터와 진단 정보를 저장하는 함수
def save_case_samples(fold, val_data, case_dir, metrics):
    case_map = {1: 'TP', 2: 'FP', 3: 'FN', 4: 'TN'}
    all_case_empty = True

    for code, case_word in case_map.items():
        case_df = val_data[val_data['case'] == code]
        if not case_df.empty:
            all_case_empty = False
            case_path = os.path.join(case_dir, f'case_{case_word}.csv')
            case_df.to_csv(case_path, index=False, encoding='utf-8')
            print(f"Case {case_word} 샘플 {len(case_df)}개 저장 완료: {case_path}")
        else:
            print(f"Case {case_word} 샘플 없음: 저장 생략")

    if all_case_empty:
        print(f"\n[경고] Fold {fold}: 모든 케이스가 0개입니다. 데이터셋 자체에 문제가 있을 수 있습니다.")
        print(f"정답값 분포: {np.bincount(metrics['all_labels']) if len(metrics['all_labels']) > 0 else '없음'}")
        print(f"예측값 분포: {np.bincount(metrics['all_preds']) if len(metrics['all_preds']) > 0 else '없음'}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        full_save_path = os.path.join(case_dir, f'fold_{fold}_all_cases_empty.csv')
        val_data.to_csv(full_save_path, index=False, encoding='utf-8')
        print(f"전체 검증 데이터셋 저장 완료: {full_save_path}")

# 모델 평가 결과 출력 및 케이스별 검증 샘플 저장
def report_evaluation(fold, metrics, dataset=None, val_idx=None):
    print_evaluation_summary(fold, metrics)  # 평가 결과 출력
    plot_roc_curve(fold, metrics['all_labels'], metrics['all_probs'])  # ROC 곡선 저장

    if dataset is not None and val_idx is not None:
        true_labels = np.array(metrics['all_labels'])
        pred_labels = np.array(metrics['all_preds'])
        new_labels = classify_cases(true_labels, pred_labels)  # 케이스 분류

        val_data = dataset.data.iloc[val_idx].copy()
        val_data['prediction'] = pred_labels
        val_data['case'] = new_labels

        case_dir = f'../Result/BiLSTM/case_samples/fold_{fold}'
        os.makedirs(case_dir, exist_ok=True)

        save_case_samples(fold, val_data, case_dir, metrics)  # 케이스별 저장 + 진단

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

# train 함수
def train(model, dataloader, optimizer, scaler, fold, epoch, epochs):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    total_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader, start=1):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        token_weights = batch['token_weights'].to(device)  # token_weights 적용

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

        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        sys.stdout.write(f"\rFold {fold} | Epoch {epoch}/{epochs} | Batch {batch_idx}/{total_batches} | Batch Loss: {loss.item():.5f}")
        sys.stdout.flush()  # 즉시 출력 버퍼 비우기

    avg_loss = total_loss / len(dataloader)
    train_acc = accuracy_score(all_labels, all_preds)
    print(f"\nFold {fold} | Epoch {epoch}/{epochs} Completed | Loss: {avg_loss:.5f} | Acc: {train_acc:.4f}\n")

    return {
        'avg_loss': avg_loss,
        'accuracy': train_acc
    }

if __name__ == '__main__':
    dataset = VoicePhishingDataset()
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
        model = BiLSTMAttention(vocab_size).to(device)
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=3e-5, weight_decay=5e-5)
        early_stopping = EarlyStopping(patience=30, min_delta=0.001)

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_annealing)
        scaler = GradScaler()

        train_metrics_list = []
        val_metrics_list = []

        # 에포크마다 학습 수행
        for epoch in range(1, num_epochs + 1):
            train_metrics = train(model, train_loader, optimizer, scaler, fold, epoch, num_epochs)
            scheduler.step()

            val_metrics = evaluate(model, val_loader)

            train_metrics_list.append(train_metrics)
            val_metrics_list.append(val_metrics)

            if early_stopping(val_metrics['avg_loss']):
                report_evaluation(fold, val_metrics, dataset, val_idx)
                break

        # Fold 끝난 후 그래프 저장
        plot_metrics_from_lists(fold, train_metrics_list, val_metrics_list)

        # 평가
        fold_metrics['loss'].append(val_metrics['avg_loss'])
        fold_metrics['acc'].append(val_metrics['accuracy'])
        fold_metrics['roc_auc'].append(val_metrics['roc_auc'])
        fold_metrics['true_acc'].append(val_metrics['true_acc'])
        fold_metrics['false_acc'].append(val_metrics['false_acc'])

        # fold 모델 저장 (각 fold마다 저장)
        model_dir = '../Result/BiLSTM/model'
        os.makedirs(model_dir, exist_ok=True)
        fold_model_file = os.path.join(model_dir, f'fold_{fold}.pth')
        torch.save(model.state_dict(), fold_model_file)
        print(f"Fold {fold} 모델 저장 완료: {fold_model_file}\n")

        # 현재 fold가 최적의 성능(ROC AUC)이라면 모델을 저장 및 갱신
        if val_metrics['roc_auc'] > best_roc_auc:
            best_roc_auc = val_metrics['roc_auc']
            best_fold = fold
            best_model_file = fold_model_file

        fold += 1

    # 전체 평균 성능 출력
    print("=== Overall Metrics ===")
    for key, values in fold_metrics.items():
        print(f"Average {key.capitalize()}: {np.mean(values):.4f}")

    # 최적 모델 파일 이름 출력
    print(f"\n최적의 모델은 Fold {best_fold}이며, 모델 파일 이름은 '{best_model_file}' 입니다.")
    final_model_name = os.path.join(model_dir, 'best_model.pth')
    shutil.copy(best_model_file, final_model_name)