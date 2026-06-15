import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer


class ESMBranch(nn.Module):
    def __init__(self, model_path, max_length=15, lstm_hidden_size=64, dropout=0.2):
        super().__init__()
        self.max_length = max_length
        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        hidden_size = self.model.config.hidden_size

        self.bilstm = nn.LSTM(hidden_size, lstm_hidden_size, num_layers=1, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv2d(1, 3, kernel_size=(4, hidden_size))
        self.conv2 = nn.Conv2d(1, 3, kernel_size=(5, hidden_size))
        self.conv3 = nn.Conv2d(1, 3, kernel_size=(6, hidden_size))
        self.fc = nn.Linear(99, lstm_hidden_size * 2)

    def forward(self, sequences):
        device = next(self.parameters()).device
        tokens = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        outputs = self.model(**tokens)

        pooled = outputs.pooler_output
        last_hidden = outputs.last_hidden_state.unsqueeze(1)
        batch_size = last_hidden.size(0)
        cnn_features = torch.cat(
            (
                self.conv1(last_hidden).view(batch_size, -1),
                self.conv2(last_hidden).view(batch_size, -1),
                self.conv3(last_hidden).view(batch_size, -1),
            ),
            dim=1,
        )
        cnn_features = self.fc(cnn_features)

        lstm_features, _ = self.bilstm(pooled.unsqueeze(0))
        lstm_features = self.dropout(lstm_features).squeeze(0)
        return cnn_features, lstm_features


class ProtBertBranch(nn.Module):
    def __init__(self, model_path, max_length=15, lstm_hidden_size=64, dropout=0.2):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
        self.model = BertModel.from_pretrained(model_path)
        hidden_size = self.model.config.hidden_size

        self.bilstm = nn.LSTM(hidden_size, lstm_hidden_size, num_layers=1, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv2d(1, 3, kernel_size=(4, hidden_size))
        self.conv2 = nn.Conv2d(1, 3, kernel_size=(5, hidden_size))
        self.conv3 = nn.Conv2d(1, 3, kernel_size=(6, hidden_size))
        self.fc = nn.Linear(99, lstm_hidden_size * 2)

    def forward(self, sequences):
        device = next(self.parameters()).device
        tokens = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        outputs = self.model(**tokens)

        pooled = outputs.pooler_output
        last_hidden = outputs.last_hidden_state.unsqueeze(1)
        batch_size = last_hidden.size(0)
        cnn_features = torch.cat(
            (
                self.conv1(last_hidden).view(batch_size, -1),
                self.conv2(last_hidden).view(batch_size, -1),
                self.conv3(last_hidden).view(batch_size, -1),
            ),
            dim=1,
        )
        cnn_features = self.fc(cnn_features)

        lstm_features, _ = self.bilstm(pooled.unsqueeze(0))
        lstm_features = self.dropout(lstm_features).squeeze(0)
        return cnn_features, lstm_features


class DeepTreeAAPred(nn.Module):
    def __init__(self, esm_model_path, protbert_model_path, max_length=15):
        super().__init__()
        self.esm_branch = ESMBranch(esm_model_path, max_length=max_length)
        self.protbert_branch = ProtBertBranch(protbert_model_path, max_length=max_length)
        self.classifier = nn.Linear(128, 2)

    def forward(self, esm_sequences, protbert_sequences):
        esm_cnn, esm_lstm = self.esm_branch(esm_sequences)
        protbert_cnn, protbert_lstm = self.protbert_branch(protbert_sequences)
        fused = esm_cnn + esm_lstm + protbert_cnn + protbert_lstm
        return self.classifier(fused)

