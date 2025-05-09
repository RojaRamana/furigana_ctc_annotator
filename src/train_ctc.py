import torch
import torch.nn as nn
import torch.optim as optim
import os
from data_loader import KanjiHiraganaDataset
from ctc_model import CTCAnnotator

# Dataset
dataset = KanjiHiraganaDataset("data/kanji_hiragana_pairs.tsv")

char_vocab = list("".join([kanji + hira for kanji, hira in dataset]))
char2idx = {char: idx + 1 for idx, char in enumerate(set(char_vocab))}
char2idx["<blank>"] = 0
vocab_size = len(char2idx)

def encode(text):
    return torch.tensor([char2idx[c] for c in text], dtype=torch.long)

HIDDEN_SIZE = 256
NUM_LAYERS = 5
EPOCHS = 10
LEARNING_RATE = 0.0001

embedding = nn.Embedding(vocab_size, HIDDEN_SIZE)
model = CTCAnnotator(vocab_size, HIDDEN_SIZE, NUM_LAYERS)
criterion = nn.CTCLoss(blank=0)
optimizer = optim.Adam(list(model.parameters()) + list(embedding.parameters()), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    total_loss = 0.0
    valid_batches = 0
    for i in range(len(dataset)):
        kanji, hiragana = dataset[i]

        if len(hiragana) == 0 or len(kanji) == 0:
            continue

        # Encode sequences
        input_indices = encode(kanji).unsqueeze(0)  # (1, seq_len)
        target_seq = encode(hiragana)

        # Expand input to match or exceed target length by repeating
        repeat_factor = (len(target_seq) + input_indices.size(1) - 1) // input_indices.size(1)
        expanded_indices = input_indices.repeat(1, repeat_factor)  # Repeat
        expanded_indices = expanded_indices[:, :len(target_seq)]   # Trim to target length

        input_seq = embedding(expanded_indices)

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
