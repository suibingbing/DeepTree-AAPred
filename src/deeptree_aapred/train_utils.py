import random

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, dataloader, device):
    model.eval()
    labels = []
    scores = []

    with torch.no_grad():
        for esm_sequences, batch_labels, protbert_sequences in dataloader:
            batch_labels = batch_labels.to(device)
            logits = model(esm_sequences, protbert_sequences)
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            labels.extend(batch_labels.cpu().tolist())
            scores.extend(probabilities.cpu().tolist())

    predictions = [1 if score >= 0.5 else 0 for score in scores]
    return {
        "accuracy": accuracy_score(labels, predictions),
        "auc": roc_auc_score(labels, scores),
        "mcc": matthews_corrcoef(labels, predictions),
        "recall": recall_score(labels, predictions, zero_division=0),
        "precision": precision_score(labels, predictions, zero_division=0),
        "labels": labels,
        "scores": scores,
    }

