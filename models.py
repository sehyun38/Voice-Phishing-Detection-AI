from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as f
from typing import Optional

# GRU 모델 정의
class GRUAttention(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int):
        super(GRUAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention_fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                token_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(input_ids)  # (B, L, D)
        if token_weights is not None:
            x = x * token_weights.unsqueeze(-1)

        gru_out, _ = self.gru(x)  # (B, L, 2H)
        attn_scores = self.attention_fc(gru_out).squeeze(-1)  # (B, L)

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        if token_weights is not None:
            attn_scores = attn_scores * token_weights  # 보정

        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, L, 1)
        context = (gru_out * attn_weights).sum(dim=1)  # (B, 2H)

        context = self.dropout(context)
        return self.fc(context)

# FastText 모델 정의
class FastTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(FastTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids: torch.Tensor):
        x = self.embedding(input_ids)  # [B, L, D]
        x = x.mean(dim=1)              # [B, D] - 단어 임베딩 평균
        x = self.dropout(x)
        return self.fc(x)

# LiteAttention 모델 정의
class LiteAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids: torch.Tensor):
        x = self.embedding(input_ids)  # [B, L, D]
        Q, K, V = self.q(x), self.k(x), self.v(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)  # [B, L, L]
        context = torch.matmul(attn_weights, V)       # [B, L, D]
        pooled = context.mean(dim=1)                  # [B, D]
        return self.fc(self.dropout(pooled))

# DPCNN 모델 정의
class DPCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv_region = nn.Conv1d(embed_dim, 250, kernel_size=3, padding=1)
        self.conv = nn.Conv1d(250, 250, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(250, num_classes)

    def forward(self, input_ids: torch.Tensor):
        x = self.embedding(input_ids).transpose(1, 2)  # (B, D, L)
        x = f.relu(self.conv_region(x))
        x = f.relu(self.conv(x)) + x  # residual
        while x.size(2) > 2:
            x = self.pool(x)
            x = f.relu(self.conv(x)) + x  # residual block
        x = f.adaptive_max_pool1d(x, 1).squeeze(2)
        return self.fc(x)

# KMaxCNN 모델 정의
class KMaxCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, k=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.k = k
        self.fc = nn.Linear(128 * k, num_classes)

    def forward(self, input_ids: torch.Tensor,):
        x = self.embedding(input_ids).transpose(1, 2)  # [B, D, L]
        x = torch.relu(self.conv(x))                  # [B, C, L]
        topk = x.topk(self.k, dim=2).values           # [B, C, k]
        x = topk.view(x.size(0), -1)                  # [B, C*k]
        return self.fc(x)

# TextCNN 모델 정의
class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 커널 5개로 구성
        self.conv3 = nn.Conv1d(embed_dim, 120, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(embed_dim, 120, kernel_size=4, padding=2)
        self.conv6 = nn.Conv1d(embed_dim, 120, kernel_size=6, padding=2)
        self.conv8 = nn.Conv1d(embed_dim, 120, kernel_size=8, padding=4)

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Sequential(
            nn.Linear(4 * 120, 256),  # 4개의 conv
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids: torch.Tensor,
                token_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(input_ids)

        if token_weights is not None:
            x = x * token_weights.unsqueeze(-1)

        x = x.permute(0, 2, 1)

        x3 = torch.max(f.relu(self.conv3(x)), dim=2)[0]
        x4 = torch.max(f.relu(self.conv4(x)), dim=2)[0]
        x6 = torch.max(f.relu(self.conv6(x)), dim=2)[0]
        x8 = torch.max(f.relu(self.conv8(x)), dim=2)[0]

        x = torch.cat([x3, x4, x6, x8], dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# BiLSTM 모델 정의
class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size:int, embed_dim:int, hidden_dim:int, num_classes:int):
        super(BiLSTMAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention_fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                token_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
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

# RCNN 모델 정의
class RCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int):
        super(RCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Conv1D → 문맥 정보 추출 (입력은 [B, C, L])
        self.conv = nn.Conv1d(in_channels=hidden_dim * 2,
                              out_channels=hidden_dim,
                              kernel_size=3, padding=1)

        self.dropout = nn.Dropout(0.5)

        # 중간 표현 확장 후 출력
        self.hidden_fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.out_fc = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, input_ids: torch.Tensor,
                token_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(input_ids)  # [B, L, D]

        if token_weights is not None:
            x = x * token_weights.unsqueeze(-1)

        x, _ = self.lstm(x)  # [B, L, H*2]

        x = x.permute(0, 2, 1)  # [B, H*2, L]
        x = self.conv(x)        # [B, H, L]
        x = torch.relu(x)

        x = torch.max(x, dim=2)[0]  # [B, H]  ← Global Max Pooling

        x = self.dropout(x)
        x = f.gelu(self.hidden_fc(x))  # [B, H//2]
        x = self.out_fc(x)             # [B, num_classes]

        return x

# MHA Pooling 모듈
class MHAPooling(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super(MHAPooling, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        batch_size = sequence_output.size(0)
        query = self.query.expand(batch_size, -1, -1)
        attn_output, _ = self.attention(query, sequence_output, sequence_output)
        return attn_output.squeeze(1)

# Mean Pooling 모듈
class MeanPooling(nn.Module):
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=1)

# CLS Pooling 모듈
class CLSPooling(nn.Module):
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x[:, 0, :]

#  KLUEBertModel 모델
class KLUEBertModel(nn.Module):
    def __init__(self, pooling: str, num_classes: int):
        super(KLUEBertModel, self).__init__()
        self.bert = BertModel.from_pretrained("klue/bert-base", return_dict=False)
        hidden_size = self.bert.config.hidden_size

        if pooling == "mha":
            self.pooling = MHAPooling(hidden_size)
        elif pooling == "mean":
            self.pooling = MeanPooling()
        elif pooling == "cls":
            self.pooling = CLSPooling()
        else:
            raise ValueError("Unsupported pooling type")

        self.fc1 = nn.Linear(hidden_size, 256)
        self.norm1 = nn.LayerNorm(256)
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(256, 128)
        self.norm2 = nn.LayerNorm(128)
        self.drop2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(128, 64)
        self.norm3 = nn.LayerNorm(64)
        self.drop3 = nn.Dropout(0.2)

        self.classifier = nn.Linear(64, num_classes)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_weights: Optional[torch.Tensor] = None) -> torch.Tensor:

        if token_weights is None:
            token_weights = torch.ones_like(input_ids, dtype=torch.float)

        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]

        token_weights = token_weights * attention_mask
        sequence_output = sequence_output * token_weights.unsqueeze(-1)

        pooled_output = self.pooling(sequence_output)

        x = self.fc1(pooled_output)
        x = nn.functional.gelu(x)
        x = self.norm1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = nn.functional.gelu(x)
        x = self.norm2(x)
        x = self.drop2(x)

        x = self.fc3(x)
        x = nn.functional.gelu(x)
        x = self.norm3(x)
        x = self.drop3(x)

        return self.classifier(x)

#CNN_v2
class CNNModelV2(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, num_classes:int):
        super(CNNModelV2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(embed_size, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Dropout(0.3),

            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Dropout(0.3),

            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(0.3)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                token_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(input_ids)
        if token_weights is not None and attention_mask is not None:
            token_weights = token_weights * attention_mask
        if token_weights is not None:
            x = x * token_weights.unsqueeze(-1)
        x = x.permute(0, 2, 1)
        x = self.cnn_layers(x)
        x = x.flatten(start_dim=1)
        return self.fc_layers(x)

#CNN_v3
class CNNModelV3(nn.Module):
    def __init__(self, vocab_size:int, embed_size : int, num_classes : int):
        super(CNNModelV3, self).__init__()
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

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                token_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
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
