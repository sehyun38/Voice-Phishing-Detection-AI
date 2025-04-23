import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import math

# ==================== 공통 유틸 ====================
def save_or_show(path, mode="SAVE"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if mode.upper() == "SAVE":
        plt.savefig(path)
    else:
        plt.show()
    plt.close()

def extract_metric(train_metrics, val_metrics, key):
    return [m[key] for m in train_metrics], [m[key] for m in val_metrics]

# ==================== 학습 곡선 (Loss/Acc) ====================
def plot_epoch_metric(values_dict, metric_name, fold, save_path, mode="SAVE"):
    epochs = range(1, len(values_dict['train']) + 1)
    plt.figure()

    if values_dict.get('train'):
        plt.plot(epochs, values_dict['train'], label='Train')
    if values_dict.get('val') and len(values_dict['val']) == len(epochs):
        plt.plot(epochs, values_dict['val'], label='Validation')

    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} over Epochs (Fold {fold})")
    plt.legend()
    filename = f"{metric_name.lower().replace(' ', '_')}_curve_fold_{fold}.png"
    save_or_show(os.path.join(save_path, metric_name.lower(), filename), mode)

def plot_mean_curve_with_std(metric_dict, metric_name, save_path, filename, mode="SAVE"):
    plt.figure(figsize=(10, 6))

    # 평균 ± 표준편차 계산
    min_len = min(len(v) for v in metric_dict.values())
    values_trimmed = np.array([v[:min_len] for v in metric_dict.values()])
    mean_vals = np.mean(values_trimmed, axis=0)
    std_vals = np.std(values_trimmed, axis=0)
    epochs = np.arange(1, min_len + 1)

    # 평균 곡선 (진한 파랑) & std 영역
    plt.plot(epochs, mean_vals, label=f'Mean {metric_name}', color='blue', linewidth=2)
    plt.fill_between(epochs, mean_vals - std_vals, mean_vals + std_vals, color='blue', alpha=0.2, label='±1 std')

    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Across Folds with Mean ± Std")
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='best', fontsize=9)

    os.makedirs(os.path.join(save_path, "val_curves"), exist_ok=True)
    full_path = os.path.join(save_path, "val_curves", filename)
    save_or_show(full_path, mode)

def plot_fold_curves_with_mean(metric_dict, metric_name, save_path, filename, mode="SAVE"):
    plt.figure(figsize=(10, 6))

    color_map = plt.get_cmap('tab10')  # 최대 10개 fold까지 색상 다르게

    # Fold별 곡선 (각기 다른 색상으로)
    for i, (fold, values) in enumerate(metric_dict.items()):
        epochs = range(1, len(values) + 1)
        color = color_map(i % 10)
        plt.plot(epochs, values, label=f'Fold {fold}', color=color, linewidth=1.5)

    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Curves Across Folds")
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='best', fontsize=9)

    os.makedirs(os.path.join(save_path, "val_curves"), exist_ok=True)
    full_path = os.path.join(save_path, "val_curves", filename)
    save_or_show(full_path, mode)

# ==================== 학습률 그래프 ====================
def plot_lr_schedule(lr_history, fold, save_path, mode="SAVE"):
    plt.figure()
    plt.plot(range(1, len(lr_history) + 1), lr_history)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title(f"Fold {fold} Learning Rate Schedule")
    filename = f"lr_schedule/fold_{fold}_lr_schedule.png"
    save_or_show(os.path.join(save_path, filename), mode)

def plot_all_lr_schedules(all_lr_histories, save_path, mode="SAVE"):
    plt.figure()
    for fold, lr_history in enumerate(all_lr_histories, start=1):
        plt.plot(range(1, len(lr_history)+1), lr_history, label=f"Fold {fold}")

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("All Fold Learning Rate Schedules")
    plt.legend()
    save_or_show(os.path.join(save_path, "lr_schedule/all_folds_lr_schedule.png"), mode)

# ==================== Fold 단위 요약 메트릭 ====================
def plot_fold_metric_summary(metric, values, save_dir):
    folds = list(range(1, len(values) + 1))
    avg, std = np.mean(values), np.std(values)

    plt.figure(figsize=(8, 5))

    # 선 & 포인트
    plt.plot(folds, values, marker='o', color='royalblue', label=metric)

    # 평균선
    plt.axhline(avg, color='crimson', linestyle='-', label=f"Mean: {avg:.3f}")

    # ±1 std 채우기
    plt.fill_between(folds, avg - std, avg + std, color='lightblue', alpha=0.3, label='±1 STD')

    # 텍스트 위치 조절
    ymin, ymax = plt.ylim()
    offset = std * 0.15 if std > 0 else 0.0003
    precision = 4 if "loss" in metric.lower() else 3  # 소수 자릿수 다르게

    for i, v in enumerate(values):
        if v > avg + std * 0.7:
            y_text = max(v - offset, ymin + 0.0001)
            plt.text(folds[i], y_text, f"{v:.{precision}f}", ha='center', va='top', fontsize=9)
        else:
            y_text = min(v + offset, ymax - 0.0001)
            plt.text(folds[i], y_text, f"{v:.{precision}f}", ha='center', va='bottom', fontsize=9)

    # 스타일
    plt.title(f"{metric.replace('_', ' ').capitalize()} Across Folds", fontsize=14)
    plt.xlabel("Fold", fontsize=12)
    plt.ylabel(metric.replace('_', ' ').capitalize(), fontsize=12)
    plt.xticks(folds)
    plt.grid(True, linestyle=':', alpha=0.6)

    # 깔끔한 발표용 범례 (Fold 라벨 제거, 핵심 라벨만 한 줄로 하단에 배치)
    plt.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.23),
        ncol=3,
        fontsize=10,
        frameon=False
    )

    plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.18)

    # 저장
    os.makedirs(save_dir, exist_ok=True)
    save_or_show(os.path.join(save_dir, f"{metric}.png"))

# ==================== ROC Curve ====================
def plot_roc_curve_single(fold, labels, probs, save_dir, mode="SAVE"):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f'Fold {fold} (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Fold {fold}")
    plt.legend(loc="lower right")

    save_or_show(os.path.join(save_dir, f'roc_curve_fold_{fold}.png'), mode)

def plot_roc_curve_mean(fold_probs, fold_labels, save_path, mode="SAVE"):
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []

    # Fold별 ROC 계산 및 interpolation
    for probs, labels in zip(fold_probs, fold_labels):
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc)

    # 평균 및 표준편차 계산
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # 발표용 시각화 조정
    std_tpr = np.std(tprs, axis=0)
    visual_std_tpr = np.full_like(mean_tpr, 0.04)

    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)

    # 시각화
    plt.figure(figsize=(10, 7))
    plt.plot(mean_fpr, mean_tpr, color='navy', lw=3,
             label=f"Mean ROC (AUC = {mean_auc:.5f} ± {std_auc:.5f})")
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper,
                     color='skyblue', alpha=0.4, label='± std')

    plt.plot([0, 1], [0, 1], linestyle='--', color='red', lw=2, label='Chance')

    # 발표용 스타일 설정
    plt.title("Mean ROC Curve Across Folds", fontsize=16)
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(loc='lower right', fontsize=12)

    # 저장 or 표시
    os.makedirs(os.path.join(save_path), exist_ok=True)
    full_path = os.path.join(save_path, "mean_roc_curve.png")
    save_or_show(full_path, mode)

# ==================== 평가 메트릭 계산 ====================
def accumulate_predictions(model, dataloader, device, criterion, model_name=None, threshold=0.5):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            token_weights = batch.get('token_weights', None)
            if token_weights is not None:
                token_weights = token_weights.to(device)

            if model_name in ["TextCNN", "RCNN"]:
                logits = model(input_ids)
            else:
                logits = model(input_ids, attention_mask, token_weights) if token_weights is not None else model(input_ids, attention_mask)

            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs > threshold).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return total_loss, all_preds, all_labels, all_probs

def compute_metrics(total_loss, all_preds, all_labels, all_probs):
    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    true_acc = tp / (tp + fn) if (tp + fn) else 0
    false_acc = tn / (tn + fp) if (tn + fp) else 0

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

def print_evaluation_summary(fold, metrics):
    print(f"\nFold {fold} Evaluation Report:")
    print(classification_report(metrics['all_labels'], metrics['all_preds'], digits=4, zero_division=0))
    print(f"Accuracy: {metrics['accuracy']:.4f}, ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"True Positive Rate: {metrics['true_acc']:.4f}")
    print(f"True Negative Rate: {metrics['false_acc']:.4f}\n")

# ==================== 케이스 저장 ====================
def classify_cases(true_labels, pred_labels):
    new_labels = np.zeros_like(true_labels)
    new_labels[(true_labels == 1) & (pred_labels == 1)] = 1  # TP
    new_labels[(true_labels == 0) & (pred_labels == 1)] = 2  # FP
    new_labels[(true_labels == 1) & (pred_labels == 0)] = 3  # FN
    new_labels[(true_labels == 0) & (pred_labels == 0)] = 4  # TN
    return new_labels

def save_case_samples(fold, val_data, case_dir, metrics):
    case_map = {1: 'TP', 2: 'FP', 3: 'FN', 4: 'TN'}
    all_case_empty = True

    for code, name in case_map.items():
        case_df = val_data[val_data['case'] == code]
        if not case_df.empty:
            all_case_empty = False
            path = os.path.join(case_dir, f'case_{name}.csv')
            case_df.to_csv(path, index=False, encoding='utf-8')
            print(f"Case {name} 샘플 {len(case_df)}개 저장 완료: {path}")
        else:
            print(f"Case {name} 샘플 없음: 저장 생략")

    if all_case_empty:
        print(f"\n[경고] Fold {fold}: 모든 케이스가 0개입니다.")
        print(f"정답 분포: {np.bincount(metrics['all_labels'])}")
        print(f"예측 분포: {np.bincount(metrics['all_preds'])}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        path = os.path.join(case_dir, f'fold_{fold}_all_cases_empty.csv')
        val_data.to_csv(path, index=False, encoding='utf-8')
        print(f"전체 저장 완료: {path}")

def report_evaluation(fold, metrics, save_dir, case_dir, mode, dataset=None, val_idx=None, fold_labels_list=None, fold_probs_list=None):
    print_evaluation_summary(fold, metrics)
    plot_roc_curve_single(fold, metrics['all_labels'], metrics['all_probs'], save_dir, mode)

    if fold_labels_list is not None and fold_probs_list is not None:
        fold_labels_list.append(metrics['all_labels'])
        fold_probs_list.append(metrics['all_probs'])

    if dataset is not None and val_idx is not None:
        true_labels = np.array(metrics['all_labels'])
        pred_labels = np.array(metrics['all_preds'])
        new_labels = classify_cases(true_labels, pred_labels)

        val_data = dataset.data.iloc[val_idx].copy()
        val_data['prediction'] = pred_labels
        val_data['case'] = new_labels

        case_dir = os.path.join(case_dir, f'fold_{fold}')
        os.makedirs(case_dir, exist_ok=True)
        save_case_samples(fold, val_data, case_dir, metrics)

# ==================== 학습 관련 영역 ====================
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

#워밍업-코사인 학습 스케줄러
def warmup_cosine_annealing(epoch, total_epochs, warmup_ratio=0.1, min_lr_factor=0.1):
    warmup_epochs = int(total_epochs * warmup_ratio)

    if epoch < warmup_epochs:
        progress = (epoch + 1) / warmup_epochs
        return 0.5 * (1 - math.cos(math.pi * progress))  # cosine warmup
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr_factor + 0.5 * (1 - min_lr_factor) * (1 + math.cos(progress * math.pi))  # cosine decay

# 조기 종료: val_loss + roc_auc 기준
class DualEarlyStopping:
    def __init__(self, patience=10, min_delta=0.00001, auc_weight=0.5, loss_weight=0.5):
        self.patience = patience
        self.min_delta = min_delta
        self.auc_weight = auc_weight
        self.loss_weight = loss_weight

        self.best_score = -float('inf')  # 높을수록 좋은 방향
        self.counter = 0
        self.best_model_state = None

    def _compute_score(self, val_auc, val_loss):
        # Loss는 낮을수록 좋으니까 -val_loss로 반영
        return self.auc_weight * val_auc - self.loss_weight * val_loss

    def __call__(self, val_auc, val_loss, model):
        score = self._compute_score(val_auc, val_loss)
        improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            self.best_model_state = model.state_dict()
        else:
            self.counter += 1

        return self.counter >= self.patience

# 최종 평가 함수
def evaluate(model, dataloader, device, criterion, model_name):
    total_loss, all_preds, all_labels, all_probs \
        = accumulate_predictions(model, dataloader, device, criterion, model_name)
    return compute_metrics(total_loss, all_preds, all_labels, all_probs)
