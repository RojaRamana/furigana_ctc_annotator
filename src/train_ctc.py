import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from data_loader import KanjiHiraganaDataset
from ctc_model import CTCAnnotator

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset
dataset = KanjiHiraganaDataset("data/kanji_hiragana_pairs.tsv")

char_vocab = list("".join([kanji + hira for kanji, hira in dataset]))
char2idx = {char: idx + 1 for idx, char in enumerate(set(char_vocab))}
char2idx["<blank>"] = 0
vocab_size = len(char2idx)

def encode(text):
    return torch.tensor([char2idx[c] for c in text], dtype=torch.long)

# Collate function for dynamic padding
def collate_fn(batch):
    inputs, targets = [], []
    for kanji, hira in batch:
        input_ids = encode(kanji)
        target_ids = encode(hira)
        if input_ids.size(0) >= target_ids.size(0):  # Valid for CTC
            inputs.append(input_ids)
            targets.append(target_ids)
    if not inputs:
        return None, None

    # Pad inputs
    input_lengths = [seq.size(0) for seq in inputs]
    target_lengths = [seq.size(0) for seq in targets]
    padded_inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    padded_targets = nn.utils.rnn.pad_sequence(targets, batch_first=True)

    return padded_inputs, padded_targets, input_lengths, target_lengths

# Hyperparameters
HIDDEN_SIZE = 256
NUM_LAYERS = 5
EPOCHS = 10
LEARNING_RATE = 0.0001
BATCH_SIZE = 16

# Model, Loss, Optimizer
embedding = nn.Embedding(vocab_size, HIDDEN_SIZE).to(device)
model = CTCAnnotator(vocab_size, HIDDEN_SIZE, NUM_LAYERS).to(device)
criterion = nn.CTCLoss(blank=0)
optimizer = optim.Adam(list(model.parameters()) + list(embedding.parameters()), lr=LEARNING_RATE)

data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

for epoch in range(EPOCHS):
    total_loss = 0.0
    valid_batches = 0
    for batch in data_loader:
        if batch[0] is None:
            continue  # Skip invalid batch

        inputs, targets, input_lengths, target_lengths = batch
        inputs, targets = inputs.to(device), targets.to(device)

        # Embedding
        inputs_embedded = embedding(inputs)

        # Forward pass
        logits = model(inputs_embedded)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)

        flat_targets = torch.cat([t[:l] for t, l in zip(targets, target_lengths)])
        input_lengths_tensor = torch.tensor(input_lengths)
        target_lengths_tensor = torch.tensor(target_lengths)

        loss = criterion(log_probs, flat_targets, input_lengths_tensor, target_lengths_tensor)
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
