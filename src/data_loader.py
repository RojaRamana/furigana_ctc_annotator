import torch
from torch.utils.data import Dataset, DataLoader

class KanjiHiraganaDataset(Dataset):
    def __init__(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        self.pairs = [line.split("\t") for line in lines]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        kanji, hiragana = self.pairs[idx]
        return kanji, hiragana

# Example usage
if __name__ == "__main__":
    dataset = KanjiHiraganaDataset("../data/kanji_hiragana_pairs.tsv")
    for kanji, hiragana in dataset:
        print(f"Kanji: {kanji} -> Hiragana: {hiragana}")
