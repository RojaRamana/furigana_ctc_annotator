import torch
from torch.utils.data import Dataset, DataLoader
import sys
import io

# Fix Windows UTF-8 print issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class KanjiHiraganaDataset(Dataset):
    def __init__(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        self.pairs = []
        for line in lines:
            parts = line.split("\t")
            if len(parts) == 2:  # Only accept valid pairs
                self.pairs.append(parts)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        kanji, hiragana = self.pairs[idx]
        return kanji, hiragana

# Example usage
if __name__ == "__main__":
    dataset = KanjiHiraganaDataset("data/kanji_hiragana_pairs.tsv")
    for kanji, hiragana in dataset:
        print(f"Kanji: {kanji} -> Hiragana: {hiragana}")
