import argparse
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler

from deeptree_aapred.data import PeptideDataset
from deeptree_aapred.model import DeepTreeAAPred
from deeptree_aapred.train_utils import evaluate, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run 5-fold validation for DeepTree-AAPred.")
    parser.add_argument("--train", default="data/AAIP_135.csv", help="Training CSV with sequence,label columns.")
    parser.add_argument("--esm-model", default=os.getenv("ESM_MODEL_PATH", "models/ESM"))
    parser.add_argument("--protbert-model", default=os.getenv("PROTBERT_MODEL_PATH", "models/protbert"))
    parser.add_argument("--output-dir", default="outputs/5fold")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--max-length", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PeptideDataset(args.train, max_length=args.max_length)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    kfold = KFold(n_splits=args.folds, shuffle=False)
    for fold, (train_indices, valid_indices) in enumerate(kfold.split(dataset), start=1):
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(train_indices),
        )
        valid_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(valid_indices),
        )

        model = DeepTreeAAPred(args.esm_model, args.protbert_model, max_length=args.max_length).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        best_auc = -1.0

        for epoch in range(args.epochs):
            model.train()
            for esm_sequences, labels, protbert_sequences in train_loader:
                labels = labels.to(device)
                logits = model(esm_sequences, protbert_sequences)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            metrics = evaluate(model, valid_loader, device)
            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                torch.save(model.state_dict(), output_dir / f"fold_{fold}.pt")
            print(
                f"fold={fold} epoch={epoch + 1} "
                f"auc={metrics['auc']:.3f} "
                f"acc={metrics['accuracy']:.3f} "
                f"mcc={metrics['mcc']:.3f}"
            )

        fold_metrics.append(best_auc)

    print(f"mean_auc={np.mean(fold_metrics):.3f} std_auc={np.std(fold_metrics):.3f}")


if __name__ == "__main__":
    main()

