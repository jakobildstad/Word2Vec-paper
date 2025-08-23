import json
import numpy as np
from pathlib import Path

from utils import build_freqs, build_vocab, should_keep, tokenize

def preprocess_text(path="src/data/text8", min_count=10, subsample_t=1e-5, save_dir="src/artifacts"):
    """
    Preprocess text8 into subsampled word IDs.
    
    Args:
        path (str): path to the raw text8 file
        min_count (int): drop words with frequency < min_count
        subsample_t (float): subsampling threshold, ~1e-5
        save_dir (str): where to save outputs (ids + vocab)
    
    Returns:
        ids_sub (np.ndarray): array of word IDs (subsampled)
        stoi (dict): word -> id
        itos (list): id -> word
        counts (dict): id -> count
    """
    text = Path(path).read_text()
    tokens = tokenize(text)

    # 1) Build vocab
    stoi, itos, counts = build_vocab(tokens, min_count=min_count)

    # 2) Map to IDs
    ids = [stoi[w] for w in tokens if w in stoi]

    # 3) Relative frequencies
    total_tokens = len(ids)
    freqs = build_freqs(counts, total_tokens)

    # 4) Subsample
    ids_sub = [wid for wid in ids if should_keep(wid, freqs, t=subsample_t)]
    ids_sub = np.array(ids_sub, dtype=np.int32)

    # 5) Save results
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / "text8_ids.npy", ids_sub)
    with open(save_dir / "vocab.json", "w") as f:
        json.dump({"stoi": stoi, "itos": itos, "counts": counts}, f)

    print(f"Original tokens: {len(tokens)}")
    print(f"Kept after min_count: {len(stoi)} vocab entries")
    print(f"Tokens after subsampling: {len(ids_sub)}")
    print(f"Saved preprocessed IDs to {save_dir}/text8_ids.npy")
    return ids_sub, stoi, itos, counts

if __name__ == "__main__":
    preprocess_text()