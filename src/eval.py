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

def analogy(a, b, c, E, stoi, itos, k=5):
    """Compute analogy: a - b + c ≈ ?"""
    if any(w not in stoi for w in [a, b, c]):
        return []
    vec = E[stoi[a]] - E[stoi[b]] + E[stoi[c]]
    vec = vec / (np.linalg.norm(vec) + 1e-12)   # normalize for cosine sim
    sims = E @ vec
    ban = {stoi[a], stoi[b], stoi[c]}           # don’t return source words
    idx = sims.argsort()[-(k+len(ban)) :][::-1] # top-k
    return [itos[j] for j in idx if j not in ban][:k]

if __name__ == "__main__":
    E, stoi, itos = load_vectors()

    # nearest neighbors
    for w in ["king", "paris", "computer", "car", "bike", "motivation", "love", "norway", "city", "big", "small"]:
        print(w, "→", nearest(w, E, stoi, itos, k=10))

    # analogies
    print("\nAnalogies:")
    print("kings - king + queen →", analogy("kings", "king", "queen", E, stoi, itos, k=5))
    print("smaller - small + big →", analogy("smaller", "small", "big", E, stoi, itos, k=5))
    print("cars - car + truck →", analogy("cars", "car", "truck", E, stoi, itos, k=5))
    print("bikes - bike + town →", analogy("bikes", "bike", "town", E, stoi, itos, k=5))