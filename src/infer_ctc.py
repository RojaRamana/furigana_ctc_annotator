import torch
import torch.nn as nn
from ctc_model import CTCAnnotator
from data_loader import KanjiHiraganaDataset
from ctc_decoder import beam_search_decoder

# === Load dataset to get dummy vocab size ===
dataset = KanjiHiraganaDataset("data/kanji_hiragana_pairs.tsv")
fallback_vocab = list("".join([kanji + hira for kanji, hira in dataset]))
fallback_char2idx = {char: idx + 1 for idx, char in enumerate(set(fallback_vocab))}
fallback_char2idx["<blank>"] = 0
vocab_size = len(fallback_char2idx)

# === Model and embedding setup ===
HIDDEN_SIZE = 256
NUM_LAYERS = 5
embedding = nn.Embedding(vocab_size, HIDDEN_SIZE)
model = CTCAnnotator(vocab_size, HIDDEN_SIZE, NUM_LAYERS)

# === Load checkpoint ===
checkpoint = torch.load("models/ctc_model_checkpoint.pth", map_location=torch.device("cpu"))

# Load model state and embedding
model.load_state_dict(checkpoint["model_state_dict"])
embedding.load_state_dict(checkpoint["embedding_state_dict"])
model.eval()

# Restore vocabulary
char2idx = checkpoint["char2idx"]
idx2char = {int(v): k for k, v in char2idx.items()}

# === Encode / Decode functions ===
def encode(text):
    return torch.tensor([char2idx[c] for c in text if c in char2idx], dtype=torch.long)

def decode(indices):
    return ''.join([idx2char[i] for i in indices if i in idx2char])

# === Inference input ===
test_sentence = "日本語"
print(f"Testing input: {test_sentence}")

# Check characters exist in vocab
for c in test_sentence:
    print(f"Character '{c}' in vocab? {'Yes' if c in char2idx else 'No'}")

# Encode input
input_indices = encode(test_sentence).unsqueeze(0)  # (1, seq_len)
print("Encoded input indices:", input_indices)

# Embed input
input_seq = embedding(input_indices)  # (1, seq_len, hidden_size)

# === Forward pass ===
with torch.no_grad():
    logits = model(input_seq)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(0)  # (seq_len, vocab_size)

    print("Logits shape:", logits.shape)
    print("Top-5 token probabilities at time step 0:")
    topk_vals, topk_indices = torch.topk(log_probs[0], 5)
    for val, idx in zip(topk_vals.tolist(), topk_indices.tolist()):
        print(f"  {idx2char.get(idx, '?')} ({idx}): {val:.4f}")

    # === Beam Search Decoding ===
    pred_indices_beam = beam_search_decoder(log_probs, beam_width=10, blank_token=char2idx["<blank>"])
    pred_text_beam = decode(pred_indices_beam)

    # === Greedy Decoding ===
    pred_indices_greedy = torch.argmax(log_probs, dim=-1).tolist()
    final_pred = []
    prev = None
    for i in pred_indices_greedy:
        if i != char2idx["<blank>"] and i != prev:
            final_pred.append(i)
        prev = i
    pred_text_greedy = decode(final_pred)

# === Output ===
print(f"\nInput: {test_sentence}")
print(f"Predicted (Beam): {pred_text_beam}")
print(f"Predicted (Greedy): {pred_text_greedy}")
