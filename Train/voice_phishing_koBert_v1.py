"""
Tokenizer : koBert 최대 512개의 토큰까지 가능(360 설정)
Model : Auto(kobert)
epochs : 3
batch size : 32
csv : 1(보이스 피싱), 0(일상 대화) 비율 1:1 불용어 제거, 중요 키워드 가중치 계산 , 인코딩 utf-8(cp949)
cross-validation 사용
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 파일 경로
data_path = "../dataset/Interactive_Dataset/Interactive_VP_Dataset_kobert_360_v3.csv"
keyword_csv_path = "../dataset/phishing_words.csv"
model_dir = "../Result/model/kobert_v1"

#디바이스 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

#하이퍼 파라미커 설정
NUM_EPOCHS = 3
BATCH_SIZE = 32
MAX_LENGTH = 360

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def load_keyword_weights(csv_path):
    try:
        keyword_df = pd.read_csv(csv_path, encoding='utf-8')
    except:
        keyword_df = pd.read_csv(csv_path, encoding='cp949')
    return dict(zip(keyword_df['word'], keyword_df['weight']))

def calculate_keyword_weight(text, keyword_dict, default_weight=0.3):
    return sum(keyword_dict.get(word, default_weight) for word in text.split())

class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        keyword_weights = inputs.get("keyword_weight", torch.ones_like(labels, dtype=torch.float))
        model_inputs = {k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
        outputs = model(**model_inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        losses = loss_fct(logits, labels)
        weighted_loss = (losses * keyword_weights).mean()
        return (weighted_loss, outputs) if return_outputs else weighted_loss

# KoBERT용 커스텀 Dataset 정의 (키워드 가중치 계산 포함)
class KoBERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, keyword_dict, max_length=MAX_LENGTH):
        self.encodings = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length)
        self.labels = labels
        self.keyword_weights = [calculate_keyword_weight(text, keyword_dict) for text in texts]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        item['keyword_weight'] = torch.tensor(self.keyword_weights[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

def train_and_evaluate_fold(fold, train_idx, val_idx, texts, labels, tokenizer, keyword_dict, model_dir):
    print(f"\nFold {fold + 1}")

    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    train_dataset = KoBERTDataset(train_texts, train_labels, tokenizer, keyword_dict)
    val_dataset = KoBERTDataset(val_texts, val_labels, tokenizer, keyword_dict)

    model = AutoModelForSequenceClassification.from_pretrained(
        "monologg/kobert", num_labels=2, trust_remote_code=True
    ).to(device)

    fold_output_dir = os.path.join(model_dir, f"fold{fold + 1}")
    os.makedirs(fold_output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=os.path.join(model_dir, f"logs/fold_{fold + 1}"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        fp16=True,
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained(fold_output_dir)

    eval_result = trainer.evaluate()
    eval_result["fold"] = fold + 1
    return eval_result

def run_cross_validation(texts, labels, tokenizer, keyword_dict, model_dir, n_splits=5):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(texts, labels)):
        result = train_and_evaluate_fold(fold, train_idx, val_idx, texts, labels, tokenizer, keyword_dict, model_dir)
        fold_results.append(result)

    best_fold = min(fold_results, key=lambda x: x["eval_loss"])
    print("\n모든 Fold 학습 완료")
    print(f"최고 성능 Fold {best_fold['fold']} 입니다.")
    print(f"성능 지표: Accuracy={best_fold['eval_accuracy']:.4f}, F1={best_fold['eval_f1']:.4f}, Loss={best_fold['eval_loss']:.4f}")
    return fold_results, best_fold


if __name__ == "__main__":
    try:
        data = pd.read_csv(data_path, encoding="utf-8")
    except:
        data = pd.read_csv(data_path, encoding="cp949")

    phishing_data = data[data['label'] == 1]
    normal_data = data[data['label'] == 0]
    normal_sampled = normal_data.sample(n=len(phishing_data), random_state=42)

    balanced_data = pd.concat([phishing_data, normal_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    texts = balanced_data['transcript'].astype(str).tolist()
    labels = balanced_data['label'].astype(int).tolist()

    keyword_dict = load_keyword_weights(keyword_csv_path)

    fold_results, best_fold = run_cross_validation(texts, labels, tokenizer, keyword_dict, model_dir)


