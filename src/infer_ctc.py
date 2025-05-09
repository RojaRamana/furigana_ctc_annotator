import torch
from ctc_model import CTCAnnotator
from data_loader import KanjiHiraganaDataset
from ctc_decoder import beam_search_decoder

# Load dataset and build vocab (same as training)
dataset = KanjiHiraganaDataset("data/kanji_hiragana_pairs.tsv")
char_vocab = list("".join([kanji + hira for kanji, hira in dataset]))
char2idx = {char: idx + 1 for idx, char in enumerate(set(char_vocab))}
char2idx["<blank>"] = 0
idx2char = {idx: char for char, idx in char2idx.items()}
vocab_size = len(char2idx)

def encode(text):
    return torch.tensor([char2idx[c] for c in text], dtype=torch.long)

def decode(indices):
    return ''.join([idx2char[i] for i in indices if i in idx2char])

# Example input sentence
test_sentence = "日本語"

# Model Setup
HIDDEN_SIZE = 256
NUM_LAYERS = 5
embedding = nn.Embedding(vocab_size, HIDDEN_SIZE)
model = CTCAnnotator(vocab_size, HIDDEN_SIZE, NUM_LAYERS)

# Prepare input
input_indices = encode(test_sentence).unsqueeze(0)  # (1, seq_len)
input_seq = embedding(input_indices)               # (1, seq_len, hidden_size)

# Forward pass
logits = model(input_seq)
log_probs = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(0)  # (seq_len, vocab_size)

# Beam Search Decoding
pred_indices = beam_search_decoder(log_probs, beam_width=3, blank_token=char2idx["<blank>"])

# Convert indices to characters
pred_text = decode(pred_indices)

print(f"Input: {test_sentence}")
print(f"Predicted Hiragana: {pred_text}")
