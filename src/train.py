# train.py
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import SkipGramDataset
from models import SkipGramNS

def train_skipgram(model, data_loader, lr=0.001, steps=None, device="cpu", log_every=200):
    model.train()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    it = iter(data_loader) # iter is a python function that returns an iterator.
    # item = next(it) is the equivalent of doing for item in object_loader

    total, n = 0.0, 0
    step = 0
    
    # Create progress bar
    pbar = tqdm(total=steps, desc="Training", unit="step")
    
    while steps is None or step < steps:
        try:
            center, context, negative = next(it)
        except StopIteration:
            it = iter(data_loader)
            center, context, negative = next(it)

        center   = center.to(device, dtype=torch.long)
        context  = context.to(device, dtype=torch.long)
        negative = negative.to(device, dtype=torch.long)

        opt.zero_grad()
        loss = model(center, context, negative)
        loss.backward()
        opt.step()

        total += loss.item()
        n += 1
        step += 1
        
        # Update progress bar
        pbar.update(1)
        
        if step % log_every == 0:
            avg_loss = total/n
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
            total, n = 0.0, 0
    
    pbar.close()
    return model

def save_vectors(model, itos, out_npz="src/artifacts/embeddings.npz"):
    vecs = model.center_embeds.weight.detach().cpu().numpy().astype("float32")
    np.savez_compressed(out_npz, vectors=vecs, itos=np.array(itos, dtype=object))
    print(f"Saved vectors to {out_npz}")

if __name__ == "__main__":
    # Paths
    ids_path   = "src/artifacts/text8_ids.npy"
    vocab_path = "src/artifacts/vocab.json"

    # Load vocab counts (for dataset's negative sampler) and itos for saving later
    with open(vocab_path) as f:
        voc = json.load(f)
    counts = {int(k): int(v) for k, v in voc["counts"].items()}
    itos = voc["itos"]
    V = max(counts.keys()) + 1
    print(f"Vocab size V={V}")

    # Dataset / DataLoader
    ds = SkipGramDataset(ids_path=ids_path, vocab_counts=counts, max_window=5, num_negatives=5)
    dl = DataLoader(ds, batch_size=1024, shuffle=True, num_workers=0)

    # Model
    model = SkipGramNS(vocab_size=V, embedding_dim=100)

    # Train (e.g., ~20k steps for a quick run)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_skipgram(model, dl, lr=0.002, steps=20000, device=device, log_every=200)

    # Save vectors for inference
    save_vectors(model, itos)