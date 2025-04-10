import torch
import torch.nn as nn
from transformers import BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import TOKENIZER_NAME

# TextCNN 모델 정의
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=None,
                 dropout_prob=0.5, num_channels=100, use_batchnorm=True, use_deep_fc=True):
        super(TextCNN, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 4, 5, 6, 7]

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            conv = nn.Conv1d(embed_dim, num_channels, kernel_size=k, padding=k // 2)
            if use_batchnorm:
                self.convs.append(nn.Sequential(conv,nn.BatchNorm1d(num_channels),nn.ReLU()))
            else:
                self.convs.append(nn.Sequential(conv,nn.ReLU()))
        self.dropout = nn.Dropout(dropout_prob)
        fc_input_dim = num_channels * len(kernel_sizes)
        if use_deep_fc:
            self.fc = nn.Sequential(
                nn.Linear(fc_input_dim, fc_input_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(fc_input_dim // 2, num_classes)
            )
        else:
            self.fc = nn.Linear(fc_input_dim, num_classes)

    def forward(self, input_ids, token_weights=None):
        x = self.embedding(input_ids)  # [batch, seq_len, embed_dim]

        if token_weights is not None:
            x = x * token_weights.unsqueeze(-1)

        x = x.permute(0, 2, 1)  # [batch, embed_dim, seq_len]

        conv_outputs = []
        for conv in self.convs:
            if x.size(2) >= conv[0].kernel_size[0]:  # input length ≥ kernel
                conv_out = conv(x)
                pooled = torch.max(conv_out, dim=2)[0]
                conv_outputs.append(pooled)

        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        return self.fc(x)

# BiLSTM 모델 정의
class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
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

# RCNN 모델 정의
class RCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(RCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, num_classes )

    def forward(self, input_ids, attention_mask=None, token_weights=None):
        x = self.embedding(input_ids)

        if token_weights is not None:
            x = x * token_weights.unsqueeze(-1)

        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        x, _ = self.lstm(x)

        if attention_mask is not None:
            x, _ = pad_packed_sequence(x, batch_first=True)

        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = torch.max(x, dim=2)[0]
        x = self.dropout(x)
        return self.fc(x)

# 3. MHA Pooling 모듈
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
    def __init__(self, pooling, num_classes):
        super(KLUEBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(TOKENIZER_NAME)
        self.pooling_type = pooling
        hidden_size = self.bert.config.hidden_size

        # Pooling 방식 선택
        if pooling == 'mha':
            self.pooling = MHAPooling(hidden_size)
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

        self.out = nn.Linear(64, num_classes )

    def forward(self, input_ids, attention_mask, token_weights=None, use_token_weights = True):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        sequence_output = outputs.last_hidden_state

        # Token weights 적용 (PAD 토큰 제거)
        if use_token_weights and token_weights is not None:
            token_weights = token_weights * attention_mask
            sequence_output = sequence_output * token_weights.unsqueeze(-1)

        pooled_output = self.pooling(sequence_output)

        x = self.fc1(pooled_output)
        x = nn.GELU()(x)
        x = self.norm1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = nn.GELU()(x)
        x = self.norm2(x)
        x = self.drop2(x)

        x = self.fc3(x)
        x = nn.GELU()(x)
        x = self.norm3(x)
        x = self.drop3(x)

        logits = self.out(x)

        return logits

class CNNModelV2(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes):
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

    def forward(self, input_ids, attention_mask=None, token_weights=None):
        x = self.embedding(input_ids)
        if token_weights is not None and attention_mask is not None:
            token_weights = token_weights * attention_mask
        if token_weights is not None:
            x = x * token_weights.unsqueeze(-1)
        x = x.permute(0, 2, 1)
        x = self.cnn_layers(x)
        x = x.flatten(start_dim=1)
        return self.fc_layers(x)

class CNNModelV3(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes):
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

    def forward(self, input_ids, attention_mask=None, token_weights=None):
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