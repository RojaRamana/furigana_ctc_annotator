import torch

def beam_search_decoder(log_probs, beam_width=3, blank_token=0):
    """
    Beam Search Decoder for CTC Output
    log_probs: (seq_len, vocab_size) - log softmax probabilities
    """
    seq_len, vocab_size = log_probs.shape
    beams = [([], 0.0)]  # (sequence, log_prob)

    for t in range(seq_len):
        new_beams = []
        for seq, score in beams:
            for c in range(vocab_size):
                new_seq = seq + [c]
                new_score = score + log_probs[t, c].item()
                new_beams.append((new_seq, new_score))
        
        # Keep top beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

    # Best beam
    best_seq, _ = beams[0]

    # Collapse repeated characters and remove blanks
    collapsed_seq = []
    prev = None
    for idx in best_seq:
        if idx != prev and idx != blank_token:
            collapsed_seq.append(idx)
        prev = idx

    return collapsed_seq
