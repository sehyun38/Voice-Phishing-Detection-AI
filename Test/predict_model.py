import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import  DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from config import  FILE_PATH, PT_SAVE_PATH, DEVICE, tokenizer, VOCAB_SIZE, MAX_LENGTH, batch_size, num_classes
from models import BiLSTMAttention, TextCNN, RCNN, KLUEBertModel, CNNModelV2, CNNModelV3
from utils import VoicePhishingDataset, report_evaluation, evaluate

# 하이퍼 파라미터
embed_size = 256
hidden_size = 128
criterion = nn.CrossEntropyLoss()

# 그래프 저장 "SAVE", "SHOW"
MODE = "SAVE"
pooling = 'mha'

#토큰, 가중치 출력
DEBUG_MODE = False

#토큰 가중치 제어
USE_TOKEN_WEIGHTS = True

# 모델 구조 및 가중치 저장한 모델 로드
model_structure_load = True
file_ext = ".pt" if model_structure_load else ".pth"

model_name = "RCNN_th" # 저장할 폴더 이름
model_type = "RCNN" # 불러올 모델 종류
model_filename = "best_model" + file_ext # 블러올 모델 파일
model_path = f"../Result/{model_type}/model/{model_filename}"

#파일 저장 결로
save_dir = f'../Result/{model_name}/ROCAUC'
case_dir = f'../Result/{model_name}/case_samples'

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {model_path}")

    if model_type != "KLUEBERT":
        if model_structure_load:
            model = torch.jit.load(model_path, map_location=DEVICE)
        else:
            if model_type == "CNN_v2":
                model = CNNModelV2(VOCAB_SIZE, embed_size, num_classes)
            elif model_type == "CNN_v3":
                model = CNNModelV3(VOCAB_SIZE, embed_size, num_classes)
            elif model_type == "TextCNN":
                model = TextCNN(VOCAB_SIZE, embed_size, num_classes)
            elif model_type == "BiLSTM_v2":
                model = BiLSTMAttention(VOCAB_SIZE, embed_size, hidden_size, num_classes)
            elif model_type == "RCNN":
                model = RCNN(VOCAB_SIZE, embed_size, hidden_size, num_classes)
            elif model_type =="KLUEBERT":
                model = KLUEBertModel(pooling, num_classes)
            else:
                raise ValueError("지원하지 않는 모델 타입입니다")

            model = model.to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        return model


def main():
    dataset = VoicePhishingDataset(FILE_PATH, PT_SAVE_PATH, tokenizer, MAX_LENGTH, USE_TOKEN_WEIGHTS)
    labels = dataset.data['label'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = {'loss': [], 'acc': [], 'roc_auc': [], 'true_acc': [], 'false_acc': []}

    fold = 1
    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        print(f"--- Fold {fold} ---")
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size)

        model = load_model(model_path)

        val_metrics = evaluate(model, val_loader, DEVICE, criterion, model_name)

        # 무조건 report_evaluation 실행
        report_evaluation(fold, val_metrics, save_dir, case_dir, MODE, dataset, val_idx)

        # 평가
        fold_metrics['loss'].append(val_metrics['avg_loss'])
        fold_metrics['acc'].append(val_metrics['accuracy'])
        fold_metrics['roc_auc'].append(val_metrics['roc_auc'])
        fold_metrics['true_acc'].append(val_metrics['true_acc'])
        fold_metrics['false_acc'].append(val_metrics['false_acc'])

        fold += 1

    # 전체 평균 성능 출력
    print("=== Overall Metrics ===")
    for key, values in fold_metrics.items():
        print(f"Average {key.capitalize()}: {np.mean(values):.4f}")

if __name__ == '__main__':
    main()