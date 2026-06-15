import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset


def pad_or_truncate(sequence, max_length=15, pad_token="0"):
    if len(sequence) > max_length:
        return sequence[:max_length]
    return sequence + pad_token * (max_length - len(sequence))


def space_amino_acids(sequence):
    return " ".join(sequence)


class PeptideDataset(Dataset):
    def __init__(self, file_path, max_length=15):
        self.file_path = Path(file_path)
        self.max_length = max_length
        self.sequences, self.labels = self._read_csv(self.file_path)
        self.protbert_sequences = [space_amino_acids(seq) for seq in self.sequences]

    @staticmethod
    def _read_csv(file_path):
        sequences = []
        labels = []
        with file_path.open("r", newline="") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                sequences.append(row["sequence"])
                labels.append(int(row["label"]))
        return sequences, labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = pad_or_truncate(self.sequences[index], self.max_length)
        protbert_sequence = self.protbert_sequences[index]
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return sequence, label, protbert_sequence

