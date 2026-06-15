import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from deeptree_aapred.data import PeptideDataset
from deeptree_aapred.model import DeepTreeAAPred
from deeptree_aapred.train_utils import evaluate, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepTree-AAPred and evaluate on an independent set.")
    parser.add_argument("--train", default="data/AAIP_135.csv", help="Training CSV with sequence,label columns.")
    parser.add_argument("--test", default="data/AAIP_28.csv", help="Independent test CSV with sequence,label columns.")
    parser.add_argument("--esm-model", default=os.getenv("ESM_MODEL_PATH", "models/ESM"))
    parser.add_argument("--protbert-model", default=os.getenv("PROTBERT_MODEL_PATH", "models/protbert"))
    parser.add_argument("--output", default="outputs/independent/deeptree_aapred.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--max-length", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = PeptideDataset(args.train, max_length=args.max_length)
    test_dataset = PeptideDataset(args.test, max_length=args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = DeepTreeAAPred(args.esm_model, args.protbert_model, max_length=args.max_length).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_auc = -1.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for esm_sequences, labels, protbert_sequences in train_loader:
            labels = labels.to(device)
            logits = model(esm_sequences, protbert_sequences)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        metrics = evaluate(model, test_loader, device)
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(model.state_dict(), output_path)
        print(
            f"epoch={epoch + 1} "
            f"auc={metrics['auc']:.3f} "
            f"acc={metrics['accuracy']:.3f} "
            f"mcc={metrics['mcc']:.3f}"
        )


if __name__ == "__main__":
    main()

