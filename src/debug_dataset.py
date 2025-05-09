from data_loader import KanjiHiraganaDataset

dataset = KanjiHiraganaDataset("data/kanji_hiragana_pairs.tsv")

print("Sample data inspection:")
for idx, pair in enumerate(dataset.pairs[:10]):  # Print first 10 pairs
    print(f"Pair {idx + 1}: {pair} (Type: {type(pair)}) - Length: {len(pair)}")
