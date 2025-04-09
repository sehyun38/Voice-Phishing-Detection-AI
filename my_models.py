import torch
import torch.nn as nn
from transformers import  BertModel

embed_size = 256
num_classes = 2
max_length = 360
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = "BiLSTM"  # CNN, TextCNN, BiLSTM, RCNN, KLUEBERT
pooling = "mha"  # cls, mean, mha (KLUEBERT 전용)
USE_TOKEN_WEIGHTS = True
DEBUG_MODE = False

class CNNModel_v3(nn.Module):
    def __init__(self, vocab_size):
        super(CNNModel_v3, self).__init__()
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

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_classes=2, kernel_sizes=[3, 4, 5], num_channels=100):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_channels, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_channels * len(kernel_sizes), num_classes)

    def forward(self, input_ids, attention_mask=None, token_weights=None):
        x = self.embedding(input_ids)
        if token_weights is not None:
            x = x * token_weights.unsqueeze(-1)
        x = x.permute(0, 2, 1)
        x = [torch.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return self.fc(x)

class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=128, num_classes=2):
        super(BiLSTMAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention_fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask=None, token_weights=None):
        x = self.embedding(input_ids)
        if token_weights is not None:
            x = x * token_weights.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attention_fc(lstm_out).squeeze(-1)
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        if token_weights is not None:
            attn_scores = attn_scores * token_weights
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = (lstm_out * attn_weights).sum(dim=1)
        context = self.dropout(context)
        return self.fc(context)

class RCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=128, num_classes=2):
        super(RCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask=None, token_weights=None):
        x = self.embedding(input_ids)
        if token_weights is not None:
            x = x * token_weights.unsqueeze(-1)
        x, _ = self.lstm(x)
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

# 4. KLUE BERT 모델
class KLUEBertModel(nn.Module):
    def __init__(self, bert_model_name="klue/bert-base", num_classes=2):
        super(KLUEBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size

        if pooling == 'mha':
            self.pooling = MHAPooling(hidden_size)
        elif pooling == 'mean':
            self.pooling = lambda x: torch.mean(x, dim=1)
        elif pooling == 'cls':
            self.pooling = lambda x: x[:, 0, :]
        else:
            raise ValueError("Invalid pooling type")

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

        if USE_TOKEN_WEIGHTS and token_weights is not None:
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

        return self.out(x)
