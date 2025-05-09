import torch
import torch.nn as nn

class CTCAnnotator(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=5):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # x: (batch_size, seq_len, hidden_size)
        x = x.transpose(1, 2)  # to (batch_size, hidden_size, seq_len)
        x = self.conv(x)
        x = x.transpose(1, 2)  # back to (batch_size, seq_len, hidden_size)
        return self.fc(x)
