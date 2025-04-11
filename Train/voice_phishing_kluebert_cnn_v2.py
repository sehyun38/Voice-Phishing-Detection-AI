"""
Tokenizer : KlueBert 최대 512(360 설정)
Model : CNN
optimizer : Adam
epochs : 100
batch size : 32
조기 종료 :  5회
csv : 1(보이스 피싱), 0(일상 대화) 비율 1:1 불용어 제거, 중요 키워드 가중치 계산 , 인코딩 utf-8(cp949)
cross-validation 사용, ROCAUC
"""

import os, sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from torch.amp import GradScaler, autocast
import shutil

from config import  FILE_PATH, PT_SAVE_PATH, DEVICE, tokenizer, VOCAB_SIZE, MAX_LENGTH
from models import CNNModelV2
from utils import (VoicePhishingDataset, plot_metrics_from_lists, accumulate_predictions,
                    compute_metrics, report_evaluation, DualEarlyStopping)
#모델 이름
model_name = "CNN_v2"

# 하이퍼 파라미터 설정
batch_size = 32
num_epochs = 100
embed_size = 256
num_classes = 2
model_structure_save = True          #모델 구조 저장
criterion = nn.CrossEntropyLoss()

# 그래프 저장 "SAVE", "SHOW"
MODE = "SAVE"

#토큰, 가중치 출력
DEBUG_MODE = False

#토큰 가중치 제어
USE_TOKEN_WEIGHTS = True

#파일 저장 결로
save_path = f'../Result/{model_name}'
save_dir = f'../Result/{model_name}/ROCAUC'
case_dir = f'../Result/{model_name}/case_samples'
model_dir = f'../Result/{model_name}/model'

# 최종 평가 함수
def evaluate(model, dataloader):
    total_loss, all_preds, all_labels, all_probs\
        =  accumulate_predictions(model, dataloader, DEVICE, criterion)
    return compute_metrics(total_loss, all_preds, all_labels, all_probs)

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

        sys.stdout.write(f"\rFold {fold} | Epoch {epoch}/{epochs} | Batch {batch_idx}/{total_batches} | Loss: {loss.item():.5f}")
        sys.stdout.flush()

    avg_loss = total_loss / len(dataloader)
    train_acc = accuracy_score(all_labels, all_preds)
    print(f"\nFold {fold} | Epoch {epoch} Completed | Loss: {avg_loss:.5f} | Acc: {train_acc:.4f}\n")

    return {
        'avg_loss': avg_loss,
        'accuracy': train_acc
    }

def main():
    dataset = VoicePhishingDataset(FILE_PATH, PT_SAVE_PATH, tokenizer, MAX_LENGTH, USE_TOKEN_WEIGHTS)
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

        model = CNNModelV2(VOCAB_SIZE, embed_size, num_classes).to(DEVICE)

        optimizer = optim.Adam(model.parameters(), lr= 2e-5, weight_decay=5e-5)
        early_stopping = DualEarlyStopping(patience=30, min_delta=0.0001)

        scaler = GradScaler()

        train_metrics_list = []
        val_metrics_list = []

        # 에포크 학습 수행
        for epoch in range(1, num_epochs + 1):
            train_metrics = train(model, train_loader, optimizer, scaler, fold, epoch, num_epochs)
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

        os.makedirs(model_dir, exist_ok=True)

        if model_structure_save:
            fold_model_file = os.path.join(model_dir, f'fold_{fold}.pt')
            scripted_model = torch.jit.script(model)
            torch.jit.save(scripted_model, fold_model_file)
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

if __name__ == '__main__':
    main()