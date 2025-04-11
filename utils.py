import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report

# 데이터셋 클래스 정의
class VoicePhishingDataset(Dataset):
    def __init__(self, file_path, keyword_path, tokenizer, max_length,
                 use_token_weights, use_precomputed_weights=True):
        self.data = pd.read_csv(file_path, encoding='utf-8') if os.path.exists(file_path) else pd.read_csv(file_path,
                                                                                                           encoding='cp949')

        # 클래스 균형 맞추기 (보이스피싱 1, 정상 대화 0)
        phishing = self.data[self.data['label'] == 1]
        normal = self.data[self.data['label'] == 0].sample(n=len(phishing), random_state=42)
        self.data = pd.concat([phishing, normal]).sample(frac=1, random_state=42).reset_index(drop=True)

        # 토큰 가중치 불러오기
        self.precomputed_weights = torch.load(
            keyword_path) if use_precomputed_weights and use_token_weights else None

        self.samples = []
        for i, (idx, row) in enumerate(self.data.iterrows()):
            text = str(row['transcript'])
            encoded = tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)

            # 사전 계산된 토큰 가중치 또는 1.0
            token_weights = self.precomputed_weights.get(idx, torch.ones_like(input_ids,
                                                                              dtype=torch.float)) if self.precomputed_weights else torch.ones_like(
                input_ids, dtype=torch.float)

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

#loss, accuracy 그래프 함수
def plot_metrics_from_lists(fold, train_metrics, val_metrics, save_path, MODE="SAVE"):
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
def plot_roc_curve(fold, all_labels, all_probs, save_dir, MODE="SAVE"):
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
def accumulate_predictions(model, dataloader,DEVICE, criterion, model_name = None):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            if model_name == "TextCNN" or model_name == "RCNN":
                logits = model(input_ids)
            else:
                logits = model(input_ids, attention_mask)

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
    print(classification_report(metrics['all_labels'], metrics['all_preds'], digits=4,zero_division=0))
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
def report_evaluation(fold, metrics, save_dir, case_dir, MODE, dataset=None, val_idx=None):
    print_evaluation_summary(fold, metrics)  # 평가 결과 출력
    plot_roc_curve(fold, metrics['all_labels'], metrics['all_probs'], save_dir, MODE)  # ROC 곡선 저장

    if dataset is not None and val_idx is not None:
        true_labels = np.array(metrics['all_labels'])
        pred_labels = np.array(metrics['all_preds'])
        new_labels = classify_cases(true_labels, pred_labels)  # 케이스 분류

        val_data = dataset.data.iloc[val_idx].copy()
        val_data['prediction'] = pred_labels
        val_data['case'] = new_labels

        case_dir = os.path.join(case_dir, f'fold_{fold}')
        os.makedirs(case_dir, exist_ok=True)

        save_case_samples(fold, val_data, case_dir, metrics)  # 케이스별 저장 + 진단

# 조기 종료: val_loss + roc_auc 기준
class DualEarlyStopping:
    def __init__(self, patience=10, min_delta=0.00001, focus='balanced'):
        self.patience = patience
        self.min_delta = min_delta
        self.focus = focus

        self.best_loss = float('inf')
        self.best_auc = -float('inf')
        self.counter = 0
        self.best_model_state = None  # 최적 가중치 저장용

    def __call__(self, val_auc, val_loss, model):
        improved = False

        if self.focus == 'loss':
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                improved = True
        elif self.focus == 'auc':
            if val_auc > self.best_auc + self.min_delta:
                self.best_auc = val_auc
                improved = True
        elif self.focus == 'balanced':
            if (val_auc > self.best_auc + self.min_delta) or (val_loss < self.best_loss - self.min_delta):
                improved = True
                self.best_auc = max(self.best_auc, val_auc)
                self.best_loss = min(self.best_loss, val_loss)

        if improved:
            self.counter = 0
            self.best_model_state = model.state_dict()  # 💾 최적 가중치 저장
        else:
            self.counter += 1

        return self.counter >= self.patience