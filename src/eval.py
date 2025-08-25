# eval_nearest.py
import numpy as np

def load_vectors(path="src/artifacts/embeddings.npz"):
    data = np.load(path, allow_pickle=True)
    E = data["vectors"]          # (V, d)
    itos = data["itos"].tolist() # list[str]
    stoi = {w:i for i,w in enumerate(itos)}
    # normalize once for cosine
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    E = E / norms
    return E, stoi, itos

def nearest(word, E, stoi, itos, k=10):
    if word not in stoi:
        return []
    v = E[stoi[word]]                   # (d,)
    sims = E @ v                        # cosine since E is normalized
    idx = np.argpartition(-sims, k+1)[:k+1]
    idx = idx[np.argsort(-sims[idx])]
    out = [itos[i] for i in idx if itos[i] != word]
    return out[:k]

if __name__ == "__main__":
    E, stoi, itos = load_vectors()
    for w in ["king", "paris", "computer"]:
        print(w, "→", nearest(w, E, stoi, itos, k=10))