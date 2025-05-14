import torch
import torch.nn as nn
import Levenshtein
from data_loader import KanjiHiraganaDataset
from ctc_model import CTCAnnotator
from ctc_decoder import beam_search_decoder

# Load checkpoint
checkpoint = torch.load("models/ctc_model_checkpoint.pth", map_location=torch.device("cpu"))

# Restore vocab + model
char2idx = checkpoint["char2idx"]
idx2char = {int(v): k for k, v in char2idx.items()}
vocab_size = len(char2idx)

def encode(text):
    return torch.tensor([char2idx[c] for c in text if c in char2idx], dtype=torch.long)

def decode(indices):
    return ''.join([idx2char[i] for i in indices if i in idx2char])

def calculate_cer(pred, target):
    return Levenshtein.distance(pred, target) / max(1, len(target))

# Model + Embedding setup
HIDDEN_SIZE = 256
NUM_LAYERS = 5
embedding = nn.Embedding(vocab_size, HIDDEN_SIZE)
model = CTCAnnotator(vocab_size, HIDDEN_SIZE, NUM_LAYERS)

model.load_state_dict(checkpoint["model_state_dict"])
embedding.load_state_dict(checkpoint["embedding_state_dict"])
model.eval()

# Dataset
dataset = KanjiHiraganaDataset("data/kanji_hiragana_pairs.tsv")

# Evaluation
cer_total = 0
sample_count = 0
print("🧪 Evaluation Results:")
with torch.no_grad():
    for kanji, hira in dataset:
        input_ids = encode(kanji).unsqueeze(0)
        input_embedded = embedding(input_ids)
        logits = model(input_embedded)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(0)

        pred_indices = beam_search_decoder(log_probs, beam_width=5, blank_token=char2idx["<blank>"])
        pred_text = decode(pred_indices)

        cer = calculate_cer(pred_text, hira)
        cer_total += cer
        sample_count += 1

        print(f"Input: {kanji} | Target: {hira} | Predicted: {pred_text} | CER: {cer:.2f}")

print(f"\n🔎 Overall CER: {cer_total / sample_count:.4f}")
