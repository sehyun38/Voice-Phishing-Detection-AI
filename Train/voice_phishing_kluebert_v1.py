"""
Tokenizer : KlueBert 최대 512(360 설정)
Model : Bert기반 koBert Classifier
optimizer : AdamW
epochs : 20
batch size : 32
조기 종료 :  5회
csv : 1(보이스 피싱), 0(일상 대화) 비율 1:1 불용어 제거, 중요 키워드 가중치 계산 , 인코딩 utf-8(cp949)
cross-validation 사용, ROCAUC, GradScaler, cosine-Schedule
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import  DataLoader, Subset
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from torch.amp import GradScaler, autocast
import shutil

from config import FILE_PATH, PT_SAVE_PATH, DEVICE, tokenizer, MAX_LENGTH, VOCAB_SIZE
from models import KLUEBertModel
from utils import (VoicePhishingDataset, accumulate_predictions, plot_metrics_from_lists,
                    compute_metrics, report_evaluation,DualEarlyStopping)

#모델 이름
model_name = "kluebert_v1"

# 하이퍼 파라미터
batch_size = 32
num_epochs = 20
num_classes = 2
pooling = 'mha'    # 사용할 pooling 방식: 'cls', 'mean', 'mha'

# 그래프 저장 모드 설정
MODE = "SAVE"

#토큰, 가중치 출력 제어
DEBUG_MODE = False

#토큰 가중치 제어
USE_TOKEN_WEIGHTS = True

#파일 저장 결로
save_path = f'../Result/{model_name}'
save_dir = f'../Result/{model_name}/ROCAUC'
case_dir = f'../Result/{model_name}/case_samples'
model_dir = f'../Result/{model_name}/model'

#모델 구조 저장
model_structure_save = True
criterion = nn.CrossEntropyLoss()

# 최종 평가 함수
def evaluate(model, dataloader):
    total_loss, all_preds, all_labels, all_probs \
        = accumulate_predictions(model, dataloader, DEVICE, criterion, model_name)
    return compute_metrics(total_loss, all_preds, all_labels, all_probs)

# 학습 함수
def train(model, dataloader, optimizer, scaler, fold, epoch, epochs):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    total_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader, start=1):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        token_weights = batch['token_weights'].to(DEVICE)

        optimizer.zero_grad()

        if DEVICE == "cuda":
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

    avg_loss = total_loss / len(dataloader)
    train_acc = accuracy_score(all_labels, all_preds)
    print(f"\nFold {fold} | Epoch {epoch} Completed | Loss: {avg_loss:.5f} | Acc: {train_acc:.4f}\n")

    return {
        'avg_loss': avg_loss,
        'accuracy': train_acc
    }

def main():
    dataset = VoicePhishingDataset(FILE_PATH, PT_SAVE_PATH, tokenizer, MAX_LENGTH, USE_TOKEN_WEIGHTS, use_precomputed_weights=True)
    labels = dataset.data['label'].values

    if DEBUG_MODE:
        lengths = [len(tokenizer.encode(text, max_length=MAX_LENGTH, truncation=False)) for text in dataset.data['transcript']]
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

        model = KLUEBertModel(pooling, num_classes).to(DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=4e-6 , weight_decay=1e-5)
        early_stopping = DualEarlyStopping(patience=30, min_delta=0.0001, focus="loss")

        total_steps = len(train_loader) * num_epochs
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1),  # 워밍업 10%
            num_training_steps=total_steps)
        scaler = GradScaler()

        train_metrics_list = []
        val_metrics_list = []

        # 에포크 학습 수행
        for epoch in range(1, num_epochs + 1):
            train_metrics = train(model, train_loader, optimizer, scaler, fold, epoch, num_epochs)
            scheduler.step()

            val_metrics = evaluate(model, val_loader)

            train_metrics_list.append(train_metrics)
            val_metrics_list.append(val_metrics)

            if early_stopping(val_metrics['roc_auc'], val_metrics['avg_loss'],model):
                model.load_state_dict(early_stopping.best_model_state)
                break

        # 무조건 report_evaluation 실행
        report_evaluation(fold, val_metrics, save_dir, case_dir, MODE, dataset, val_idx)

        # Fold 끝난 후 그래프 저장
        plot_metrics_from_lists(fold, train_metrics_list, val_metrics_list, save_path, MODE)

        # 평가
        fold_metrics['loss'].append(val_metrics['avg_loss'])
        fold_metrics['acc'].append(val_metrics['accuracy'])
        fold_metrics['roc_auc'].append(val_metrics['roc_auc'])
        fold_metrics['true_acc'].append(val_metrics['true_acc'])
        fold_metrics['false_acc'].append(val_metrics['false_acc'])

        # fold 모델 저장 (각 fold 저장)
        os.makedirs(model_dir, exist_ok=True)

        if model_structure_save:
            fold_model_file = os.path.join(model_dir, f'fold_{fold}.pt')
            dummy_input_ids = torch.randint(0, VOCAB_SIZE, (1, MAX_LENGTH)).to(DEVICE)
            dummy_attention_mask = torch.ones((1, MAX_LENGTH), dtype=torch.long).to(DEVICE)
            traced_model = torch.jit.trace(model, (dummy_input_ids, dummy_attention_mask))
            traced_model.save(fold_model_file)
        else:
            fold_model_file = os.path.join(model_dir, f'fold_{fold}.pth')
            torch.save(model.state_dict(), fold_model_file)
        print(f"Fold {fold} 모델 저장 완료: {fold_model_file}\n")

        # 현재 fold 중 최적의 성능(ROC AUC)이라면 모델을 저장 및 갱신
        if val_metrics['roc_auc'] > best_roc_auc:
            best_roc_auc = val_metrics['roc_auc']
            best_fold = fold
            best_model_file = fold_model_file

        fold += 1

    print("=== Overall Metrics ===")
    for key, values in fold_metrics.items():
        print(f"Average {key.capitalize()}: {np.mean(values):.4f}")

    print(f"\n최적의 모델은 Fold {best_fold}이며, 모델 파일 이름은 '{best_model_file}' 입니다.")
    if model_structure_save:
        final_model_name = os.path.join(model_dir, 'best_model.pt')
    else:
        final_model_name = os.path.join(model_dir, 'best_model.pth')
    shutil.copy(best_model_file, final_model_name)

if __name__ == "__main__":
    main()