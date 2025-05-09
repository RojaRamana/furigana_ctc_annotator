import torch
import torch.nn as nn
import torch.optim as optim
import os
from data_loader import KanjiHiraganaDataset
from ctc_model import CTCAnnotator

# Dataset
dataset = KanjiHiraganaDataset("data/kanji_hiragana_pairs.tsv")

# Character vocab simulation (replace this with real vocab mapping later)
char_vocab = list("".join([kanji + hira for kanji, hira in dataset]))
char2idx = {char: idx + 1 for idx, char in enumerate(set(char_vocab))}
char2idx["<blank>"] = 0
vocab_size = len(char2idx)

def encode(text):
    return torch.tensor([char2idx[c] for c in text], dtype=torch.long)

# Hyperparameters
HIDDEN_SIZE = 256
NUM_LAYERS = 5
EPOCHS = 1  # Adjust as needed
LEARNING_RATE = 0.0001  # Safer learning rate

# Model, Loss, Optimizer
embedding = nn.Embedding(vocab_size, HIDDEN_SIZE)
model = CTCAnnotator(vocab_size, HIDDEN_SIZE, NUM_LAYERS)
criterion = nn.CTCLoss(blank=0)
optimizer = optim.Adam(list(model.parameters()) + list(embedding.parameters()), lr=LEARNING_RATE)

# Training Loop
for epoch in range(EPOCHS):
    total_loss = 0.0
    valid_batches = 0
    for i in range(100):  # Adjust as needed for larger training
        kanji, hiragana = dataset[i]

        if len(hiragana) == 0 or len(kanji) == 0:
            continue

        input_indices = encode(kanji).unsqueeze(0)
        input_seq = embedding(input_indices)
        target_seq = encode(hiragana)

        if input_seq.size(1) < target_seq.size(0):
            continue

        input_lengths = torch.tensor([input_seq.size(1)])
        target_lengths = torch.tensor([len(target_seq)])

        logits = model(input_seq)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)

        loss = criterion(log_probs, target_seq.unsqueeze(0), input_lengths, target_lengths)
        loss = torch.clamp(loss, max=100.0)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        valid_batches += 1

    if valid_batches == 0:
        print("No valid batches found.")
    else:
        print(f"Epoch {epoch + 1}/{EPOCHS}, Average Loss: {total_loss / valid_batches:.4f}")

# Save model and vocab
os.makedirs("models", exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'embedding_state_dict': embedding.state_dict(),
    'char2idx': char2idx
}, "models/ctc_model_checkpoint.pth")

print("Model saved to models/ctc_model_checkpoint.pth")
