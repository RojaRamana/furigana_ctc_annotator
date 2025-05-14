import streamlit as st
import torch
import torch.nn as nn
from ctc_model import CTCAnnotator
from data_loader import KanjiHiraganaDataset
from ctc_decoder import beam_search_decoder

# Load model checkpoint
checkpoint = torch.load("models/ctc_model_checkpoint.pth", map_location="cpu")
char2idx = checkpoint["char2idx"]
idx2char = {int(v): k for k, v in char2idx.items()}
vocab_size = len(char2idx)

# Model + Embedding setup
HIDDEN_SIZE = 256
NUM_LAYERS = 5
embedding = nn.Embedding(vocab_size, HIDDEN_SIZE)
model = CTCAnnotator(vocab_size, HIDDEN_SIZE, NUM_LAYERS)
model.load_state_dict(checkpoint["model_state_dict"])
embedding.load_state_dict(checkpoint["embedding_state_dict"])
model.eval()

# Encoding/Decoding
def encode(text):
    return torch.tensor([char2idx[c] for c in text if c in char2idx], dtype=torch.long)

def decode(indices):
    return ''.join([idx2char[i] for i in indices if i in idx2char])

# Streamlit UI
st.title("🎌 Kanji to Hiragana Converter (CTC Model)")
input_text = st.text_input("Enter Japanese Kanji text:", "")

if st.button("Convert to Hiragana"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with torch.no_grad():
            input_ids = encode(input_text).unsqueeze(0)
            embedded = embedding(input_ids)
            logits = model(embedded)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(0)
            pred_indices = beam_search_decoder(log_probs, beam_width=5, blank_token=char2idx["<blank>"])
            output_text = decode(pred_indices)
        st.success(f"Predicted Hiragana: {output_text}")
