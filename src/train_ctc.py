import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import KanjiHiraganaDataset
from ctc_model import CTCAnnotator

# Dataset
dataset = KanjiHiraganaDataset("data/kanji_hiragana_pairs.tsv")

# Character vocab simulation (you'll replace this with real vocab mapping later)
char_vocab = list("".join([kanji + hira for kanji, hira in dataset]))
char2idx = {char: idx + 1 for idx, char in enumerate(set(char_vocab))}
char2idx["<blank>"] = 0
vocab_size = len(char2idx)

def encode(text):
    return torch.tensor([char2idx[c] for c in text], dtype=torch.long)

# Hyperparameters
HIDDEN_SIZE = 256
NUM_LAYERS = 5
EPOCHS = 1  # Reduced to 1 for faster testing
LEARNING_RATE = 0.001

# Model, Loss, Optimizer
embedding = nn.Embedding(vocab_size, HIDDEN_SIZE)
model = CTCAnnotator(vocab_size, HIDDEN_SIZE, NUM_LAYERS)
criterion = nn.CTCLoss(blank=0)
optimizer = optim.Adam(list(model.parameters()) + list(embedding.parameters()), lr=LEARNING_RATE)

# Training Loop
for epoch in range(EPOCHS):
    total_loss = 0.0
    for kanji, hiragana in dataset[:100]:  # Limit to first 100 samples for faster testing
        # Encode sequences
        input_indices = encode(kanji).unsqueeze(0)  # (1, seq_len)
        input_seq = embedding(input_indices)       # (1, seq_len, hidden_size)
        target_seq = encode(hiragana)

        input_lengths = torch.tensor([input_seq.size(1)])
        target_lengths = torch.tensor([len(target_seq)])

        # Forward
        logits = model(input_seq)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)  # For CTC Loss

        # Compute loss
        loss = criterion(log_probs, target_seq.unsqueeze(0), input_lengths, target_lengths)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss:.4f}")
