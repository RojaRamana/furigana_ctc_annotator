import torch
import torch.nn as nn
import torch.optim as optim
from ctc_model import CTCAnnotator

# Example hyperparameters
VOCAB_SIZE = 100  # Replace with actual vocab size
HIDDEN_SIZE = 256
NUM_LAYERS = 5
EPOCHS = 10
LEARNING_RATE = 0.001

# Model, Loss, Optimizer
model = CTCAnnotator(VOCAB_SIZE, HIDDEN_SIZE, NUM_LAYERS)
criterion = nn.CTCLoss(blank=0)  # You can define blank token id
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Example dummy data
input_seq = torch.randn(4, 50, HIDDEN_SIZE)  # (batch, seq_len, hidden_size)
target_seq = torch.randint(1, VOCAB_SIZE, (4, 30))  # (batch, target_len)

input_lengths = torch.full((4,), 50, dtype=torch.long)
target_lengths = torch.full((4,), 30, dtype=torch.long)

# Forward pass
logits = model(input_seq)  # (batch, seq_len, vocab_size)
log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
log_probs = log_probs.transpose(0, 1)  # (seq_len, batch, vocab_size)

# Compute loss
loss = criterion(log_probs, target_seq, input_lengths, target_lengths)

# Backward pass
loss.backward()
optimizer.step()

print(f"Training step completed. Loss: {loss.item():.4f}")
