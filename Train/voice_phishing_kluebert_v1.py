"""
Tokenizer : KlueBert 최대 512(360 설정)
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
from transformers import BertModel, AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from torch.amp import GradScaler, autocast
import shutil

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습 관련 설정
batch_size = 32
num_epochs = 20
num_classes = 2
pooling = 'mha'           # 사용할 pooling 방식: 'cls', 'mean', 'mha'

# tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base", trust_remote_code=True)

# 데이터 파일 경로
file_path = '../dataset/Interactive_Dataset/Interactive_VP_Dataset_kluebert_360_v1.csv'
precomputed_weights_path = '../token_weight/token_weights_kluebert.pt'

# 그래프 저장 모드 설정
MODE = "SAVE"

#토큰, 가중치 출력 제어
DEBUG_MODE = False

#토큰, 가중치 출력
DEBUG_MODE = False

#토큰 가중치 제어
USE_TOKEN_WEIGHTS = True
MAX_LENGTH = 360

# 커스텀 데이터셋 클래스 정의
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

# MHA pooling 모듈 정의 (학습 가능한 Query 사용)
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

# KoBERT 기반 모델 정의 (다양한 pooling 선택 가능)
class KLUEBertModel(nn.Module):
    def __init__(self, bert_model_name="klue/bert-base", num_classes=2):
        super(KLUEBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.pooling_type = pooling
        hidden_size = self.bert.config.hidden_size

        # Pooling 방식 선택
        if pooling == 'mha':
            self.pooling = MHAPooling(hidden_size=hidden_size)
        elif pooling == 'mean':
            self.pooling = lambda x: torch.mean(x, dim=1)
        elif pooling == 'cls':
            self.pooling = lambda x: x[:, 0, :]
        else:
            raise ValueError("Unsupported pooling type")

        # 레이어 분리
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

        # Token weights 적용 (PAD 토큰 제거)
        if USE_TOKEN_WEIGHTS and token_weights is not None:
            token_weights = token_weights * attention_mask
            sequence_output = sequence_output * token_weights.unsqueeze(-1)

        pooled_output = self.pooling(sequence_output)

        if DEBUG_MODE:
            print(f"[DEBUG] Pooled Output → mean: {pooled_output.mean().item():.4f}, std: {pooled_output.std().item():.4f}")

        x = self.fc1(pooled_output)
        if DEBUG_MODE:
            print(f"[DEBUG] FC1 → shape: {x.shape}, mean: {x.mean().item():.4f}")

        x = nn.GELU()(x)
        x = self.norm1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        if DEBUG_MODE:
            print(f"[DEBUG] FC2 → shape: {x.shape}, mean: {x.mean().item():.4f}")

        x = nn.GELU()(x)
        x = self.norm2(x)
        x = self.drop2(x)

        x = self.fc3(x)
        if DEBUG_MODE:
            print(f"[DEBUG] FC3 → shape: {x.shape}, mean: {x.mean().item():.4f}")

        x = nn.GELU()(x)
        x = self.norm3(x)
        x = self.drop3(x)

        logits = self.out(x)
        if DEBUG_MODE:
            print(f"[DEBUG] Output Logits → shape: {logits.shape}, mean: {logits.mean().item():.4f}\\n")

        return logits

#loss, accuracy 그래프 함수
def plot_metrics_from_lists(fold,train_metrics, val_metrics, save_path='../Result/kluebert_v1'):
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

# ROC 곡선 그리기 함수
def plot_roc_curve(fold, all_labels, all_probs, save_dir='../Result/kluebert_v1/ROCAUC'):
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

        case_dir = f'../Result/kluebert_v1/case_samples/fold_{fold}'
        os.makedirs(case_dir, exist_ok=True)

        save_case_samples(fold, val_data, case_dir, metrics)  # 케이스별 저장 + 진단

# 평가 함수
def evaluate(model, dataloader):
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
def train(model, dataloader, optimizer, scheduler, scaler, fold, epoch, epochs):
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

        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        sys.stdout.write(f"\rFold {fold} | Epoch {epoch}/{epochs} | Batch {batch_idx}/{total_batches} | Loss: {loss.item():.7f}")
        sys.stdout.flush()

    avg_loss = total_loss / len(dataloader)
    train_acc = accuracy_score(all_labels, all_preds)
    print(f"\nFold {fold} | Epoch {epoch} Completed | Loss: {avg_loss:.5f} | Acc: {train_acc:.4f}\n")

    return {
        'avg_loss': avg_loss,
        'accuracy': train_acc
    }

if __name__ == "__main__":
    # use_precomputed_weights 옵션 True로 설정하여 미리 계산된 토큰 가중치 사용
    dataset = VoicePhishingDataset(use_precomputed_weights=True)
    labels = dataset.data['label'].values


    if DEBUG_MODE:
        lengths = [len(tokenizer.encode(text, max_length=512, truncation=False)) for text in dataset.data['transcript']]
        plt.hist(lengths, bins=50)
        plt.xlabel("Token Length")
        plt.ylabel("Count")
        plt.title("Token Length Distribution")
        plt.show()

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

        model = KLUEBertModel().to(device)
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(model.parameters(), lr=4e-6 , weight_decay=1e-5)
        early_stopping = EarlyStopping(patience=3, min_delta=0.001)

        total_steps = len(train_loader) * num_epochs
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1),  # 워밍업 10%
            num_training_steps=total_steps)
        scaler = GradScaler()

        train_metrics_list = []
        val_metrics_list = []

        # 에포크마다 학습 수행
        for epoch in range(1, num_epochs + 1):
            train_metrics = train(model, train_loader, optimizer, scheduler, scaler, fold, epoch,num_epochs)
            val_metrics = evaluate(model, val_loader)

            train_metrics_list.append(train_metrics)
            val_metrics_list.append(val_metrics)

            if early_stopping(val_metrics['avg_loss']):
                report_evaluation(fold, val_metrics, dataset, val_idx)
                break

        # 평가
        fold_metrics['loss'].append(val_metrics['avg_loss'])
        fold_metrics['acc'].append(val_metrics['accuracy'])
        fold_metrics['roc_auc'].append(val_metrics['roc_auc'])
        fold_metrics['true_acc'].append(val_metrics['true_acc'])
        fold_metrics['false_acc'].append(val_metrics['false_acc'])

        model_dir = '../Result/kluebert_v1/model'
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

    print("=== Overall Metrics ===")
    for key, values in fold_metrics.items():
        print(f"Average {key.capitalize()}: {np.mean(values):.4f}")

    print(f"\n최적의 모델은 Fold {best_fold}이며, 모델 파일 이름은 '{best_model_file}' 입니다.")
    final_model_name = os.path.join(model_dir, 'best_model.pth')
    shutil.copy(best_model_file, final_model_name)