"""
Tokenizer : KlueBert 최대 512(360 설정)
Model : BiLSTM
optimizer : Adam
epochs : 20
batch size : 32
조기 종료 :  30회
csv : 1(보이스 피싱), 0(일상 대화) 비율 1:1 불용어 제거, 중요 키워드 가중치 계산 , 인코딩 utf-8(cp949)
cross-validation 사용, ROCAUC, 학습 스케줄러(warmup_cosine_annealing), GradScaler
"""
import os, sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import  DataLoader, Subset, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from torch.amp import GradScaler, autocast
import shutil
import math
import inspect
import argparse

from config import  FILE_PATH, PT_SAVE_PATH, DEVICE, tokenizer, VOCAB_SIZE, MAX_LENGTH, batch_size, num_classes, n_splits
from models import BiLSTMAttention, TextCNN, RCNN, KLUEBertModel,GRUAttention, FastTextClassifier,  LiteAttentionClassifier, DPCNN, KMaxCNN
from utils import (
    report_evaluation, compute_metrics,
    plot_epoch_metric, plot_fold_metric_summary, plot_mean_curve_with_std,
    plot_roc_curve_mean,  extract_metric, plot_lr_schedule, plot_fold_curves_with_mean,
)

embed_size = 256    # 임베딩 사이즈
hidden_size = 128   # 은닉층 사이즈
criterion = nn.CrossEntropyLoss()

# 그래프 저장 "SAVE", "SHOW"
MODE = "SAVE"
pooling = "MHA"

model_list = ["BiLSTM","TextCNN","RCNN","kluebert","DPCNN",
                  "GRU","FastText","LiteAttention","KMaxCNN"]

model_keys = [
    {
        "model_name": "BiLSTMAttention",
        "model_class": BiLSTMAttention,
         "init_args": {
             "vocab_size": VOCAB_SIZE,
             "embed_dim": embed_size,
             "hidden_dim": hidden_size,
             "num_classes": num_classes
         }
     },
    {
        "model_name": "TextCNN",
        "model_class": TextCNN,
        "init_args": {
             "vocab_size": VOCAB_SIZE,
             "embed_dim": embed_size,
             "num_classes": num_classes
         }
    },
    {
        "model_name": "RCNN",
        "model_class": RCNN,
        "init_args": {
            "vocab_size": VOCAB_SIZE,
            "embed_dim": embed_size,
            "hidden_dim": hidden_size,
            "num_classes": num_classes
        }
    },
    {
        "model_name": "KLUEBertModel",
        "model_class": KLUEBertModel,
        "init_args": {
            "pooling": pooling,
            "num_classes": num_classes
        }
    },
    {
        "model_name": "GRUAttention",
        "model_class": GRUAttention,
        "init_args": {
            "vocab_size": VOCAB_SIZE,
            "embed_dim": embed_size,
            "hidden_dim": hidden_size,
            "num_classes": num_classes
        }
    },
    {
        "model_name": "FastTextClassifier",
        "model_class": FastTextClassifier,
        "init_args": {
            "vocab_size": VOCAB_SIZE,
            "embed_dim": embed_size,
            "num_classes": num_classes
        }
    },
    {
        "model_name": "LiteAttentionClassifier",
        "model_class": LiteAttentionClassifier,
        "init_args": {
            "vocab_size": VOCAB_SIZE,
            "embed_dim": embed_size,
            "num_classes": num_classes
        }
    },
    {
        "model_name": "DPCNN",
        "model_class": DPCNN,
        "init_args": {
            "vocab_size": VOCAB_SIZE,
            "embed_dim": embed_size,
            "num_classes": num_classes
        }
    },
    {
        "model_name": "KMaxCNN",
        "model_class": KMaxCNN,
        "init_args": {
            "vocab_size": VOCAB_SIZE,
            "embed_dim": embed_size,
            "num_classes": num_classes
        }
    }
]

model_configs = {
    "BiLSTM": {"lr": 2e-5, "epochs": 100},
    "TextCNN": {"lr": 2e-5, "epochs": 100},
    "RCNN": {"lr": 2e-5, "epochs": 100},
    "kluebert": {"lr": 8e-6, "epochs": 20},
    "DPCNN": {"lr": 2e-5, "epochs": 40},
    "GRU": {"lr": 2e-5, "epochs": 100},
    "FastText": {"lr": 2e-5, "epochs": 30},
    "LiteAttention": {"lr": 5e-4, "epochs": 100},
    "KMaxCNN": {"lr": 2e-5, "epochs": 45}
}

# 토큰, 가중치 출력
DEBUG_MODE = False
# 토큰 가중치 제어
USE_TOKEN_WEIGHTS = True
# 모델 구조 저장
model_structure_save = True

# 데이터셋 클래스 정의
class VoicePhishingDataset(Dataset):
    def __init__(self, file_path, keyword_path,
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
            encoded = tokenizer(text, padding='max_length', truncation=True, max_length=MAX_LENGTH, return_tensors='pt')
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

def accumulate_predictions(model, dataloader, device, criterion, threshold=0.5):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    # 모델 forward가 받는 인자 이름 리스트 추출
    sig = inspect.signature(model.forward)
    forward_params = list(sig.parameters.keys())

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            # 공통 인자
            inputs = {
                "input_ids": input_ids,
            }

            # 선택 인자 추가
            if "attention_mask" in forward_params:
                inputs["attention_mask"] = batch["attention_mask"].to(DEVICE)
            if "token_weights" in forward_params:
                inputs["token_weights"] = batch.get('token_weights', None).to(device) # token_weights 적용

            logits = model(**inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs > threshold).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return total_loss, all_preds, all_labels, all_probs

# 최종 평가 함수
def evaluate(model, dataloader, device):
    total_loss, all_preds, all_labels, all_probs \
        = accumulate_predictions(model, dataloader, device, criterion,)
    return compute_metrics(total_loss, all_preds, all_labels, all_probs)

# 모델 생성 함수
def get_model_instance_by_name(name_input, model_list, device):
    name_input = name_input.lower()

    for model_info in model_list:
        model_name = model_info["model_name"].lower()
        if name_input in model_name:  # 부분 일치 & 소문자 비교
            model_class = model_info["model_class"]
            init_args = model_info["init_args"]
            return model_class(**init_args).to(device)

    raise ValueError(f"'{name_input}'에 해당하는 모델을 찾을 수 없습니다.")

# train 함수
def train(model, dataloader, optimizer, scaler, fold, epoch, epochs):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    total_batches = len(dataloader)

    # 모델 forward가 받는 인자 이름 리스트 추출
    sig = inspect.signature(model.forward)
    forward_params = list(sig.parameters.keys())

    for batch_idx, batch in enumerate(dataloader, start=1):
        input_ids = batch['input_ids'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        optimizer.zero_grad()

        # 공통 인자
        inputs = {
            "input_ids": input_ids,
        }

        # 선택 인자 추가
        if "attention_mask" in forward_params:
            inputs["attention_mask"] = batch["attention_mask"].to(DEVICE)
        if "token_weights" in forward_params:
            inputs["token_weights"] = batch["token_weights"].to(DEVICE)  # token_weights 적용

        if DEVICE == "cuda":
            with autocast(device_type='cuda'):
                logits = model(**inputs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

        else:
            logits = model(**inputs)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

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

def main(matched_model, config):
    dataset = VoicePhishingDataset(FILE_PATH, PT_SAVE_PATH, tokenizer, USE_TOKEN_WEIGHTS)
    labels = dataset.data['label'].values

    if DEBUG_MODE:
        print(f"모델: {matched_model}")
        print(f"학습률: {config['lr']}")
        print(f"에폭 수: {config['epochs']}")

        lengths = [len(tokenizer.encode(text, max_length=MAX_LENGTH, truncation=False)) for text in dataset.data['transcript']]
        plt.hist(lengths, bins=50)
        plt.xlabel("Token Length")
        plt.ylabel("Count")
        plt.title("Token Length Distribution")
        plt.show()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = {'avg_loss': [], 'accuracy': [], 'roc_auc': [], 'true_acc': [], 'false_acc': []}
    all_lr_histories = []  # 전체 fold 학습률 저장용

    val_acc_all, val_loss_all = {}, {}
    fold_probs, fold_labels = [], []

    best_roc_auc, best_fold, best_model_file = 0, None, None

    fold = 1
    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size)

        model = get_model_instance_by_name(model_name, model_keys, DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=2e-4)
        early_stopping = DualEarlyStopping(patience=20, min_delta=0.0001, auc_weight=0.6, loss_weight=0.4)

        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: warmup_cosine_annealing(epoch,config['epochs'], 0.05)
        )
        scaler = GradScaler()

        train_metrics_list, val_metrics_list, lr_history = [], [], []

        print(f"--- Fold {fold} ---")
        # 에포크 학습 수행
        for epoch in range(1, config['epochs'] + 1):
            train_metrics = train(model, train_loader, optimizer, scaler, fold, epoch, config['epochs'])
            scheduler.step()
            val_metrics = evaluate(model, val_loader, DEVICE)

            lr_history.append(optimizer.param_groups[0]['lr'])
            train_metrics_list.append(train_metrics)
            val_metrics_list.append(val_metrics)

            if early_stopping(val_metrics['roc_auc'], val_metrics['avg_loss'],model):
                model.load_state_dict(early_stopping.best_model_state)
                break

        report_evaluation(fold, val_metrics, save_dir, case_dir, MODE, dataset, val_idx, fold_labels, fold_probs)

        # 학습 곡선 저장
        train_loss, val_loss = extract_metric(train_metrics_list, val_metrics_list, 'avg_loss')
        train_acc, val_acc = extract_metric(train_metrics_list, val_metrics_list, 'accuracy')

        plot_epoch_metric({'train': train_loss, 'val': val_loss}, 'Loss', fold, save_path, MODE)
        plot_epoch_metric({'train': train_acc, 'val': val_acc}, 'Accuracy', fold, save_path, MODE)

        val_acc_all[fold] = val_acc
        val_loss_all[fold] = val_loss

        if len(val_acc_all) == n_splits:
            plot_mean_curve_with_std(val_acc_all, 'Accuracy', save_path, 'val_accuracy_all_folds.png')
            plot_mean_curve_with_std(val_loss_all, 'Loss', save_path, 'val_loss_all_folds.png')

        plot_lr_schedule(lr_history, fold, save_path, MODE)
        all_lr_histories.append(lr_history)

        # 폴드 메트릭 저장
        for k in fold_metrics:
            fold_metrics[k].append(val_metrics[k])

        # 모델 저장
        os.makedirs(model_dir, exist_ok=True)
        fold_model_file = os.path.join(model_dir, f'fold_{fold}.pt' if model_structure_save else f'fold_{fold}.pth')
        if model_structure_save:
            scripted_model = torch.jit.script(model)
            torch.jit.save(scripted_model, fold_model_file)
        else:
            torch.save(model.state_dict(), fold_model_file)

        print(f"Fold {fold} 모델 저장 완료: {fold_model_file}\n")

        if val_metrics['roc_auc'] > best_roc_auc:
            best_roc_auc = val_metrics['roc_auc']
            best_fold = fold
            best_model_file = fold_model_file

        fold += 1

    # 전체 Fold 메트릭 시각화
    for metric, values in fold_metrics.items():
        plot_fold_metric_summary(metric, values, os.path.join(save_path, "fold_metrics"))

    plot_roc_curve_mean(fold_probs, fold_labels, save_dir)

    plot_fold_curves_with_mean(val_acc_all, "Accuracy", save_path, "val_accuracy_all_folds_with_mean.png", MODE)
    plot_fold_curves_with_mean(val_loss_all, "Loss", save_path, "val_loss_all_folds_with_mean.png", MODE)

    print("=== Overall Metrics ===")
    for key, values in fold_metrics.items():
        print(f"Average {key.capitalize()}: {np.mean(values):.4f}")

    final_model_name = os.path.join(model_dir, f'best_model.{"pt" if model_structure_save else "pth"}')
    shutil.copy(best_model_file, final_model_name)
    print(f"\n최적의 모델은 Fold {best_fold}이며, '{final_model_name}'로 저장됨.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with flexible config.")
    parser.add_argument('--model', type=str, required=True, help='모델 이름 (예: gru, dp 등)')
    parser.add_argument('--lr', type=float, default=None, help='학습률 (미입력 시 모델 기본값 사용)')
    parser.add_argument('--epochs', type=int, default=None, help='에폭 수 (미입력 시 모델 기본값 사용)')

    args = parser.parse_args()

    # 모델 매칭
    model_name = args.model
    matched_model = next((m for m in model_configs if model_name.lower() in m.lower()), None)

    if matched_model is None:
        raise ValueError(f"'{model_name}'은 유효하지 않은 모델명입니다. 가능한 모델: {list(model_configs.keys())}")

    # 기본 config 불러오고 CLI로 받은 값 덮어쓰기
    config = model_configs[matched_model].copy()
    if args.lr is not None:
        config["lr"] = args.lr
    if args.epochs is not None:
        config["epochs"] = args.epochs

    # 경로 설정
    save_path = f'./Result/{matched_model}/metrics'
    save_dir = f'./Result/{matched_model}/metrics/rocauc'
    case_dir = f'./Result/{matched_model}/case_samples'
    model_dir = f'./Result/{matched_model}/model'

    # main에 값 넘겨서 학습 실행
    main(matched_model, config)