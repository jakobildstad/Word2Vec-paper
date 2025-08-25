# train.py
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import SkipGramDataset
from models import SkipGramNS
from viz import plot_losses
from eval import nearest


def train_skipgram(model, data_loader, lr=0.001, steps=None, device="cpu", log_every=200, itos=None, log_file=None):
    model.train()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr) # the "worker" that changes weights.
    # A reference to the models parameters gives access to change the requires_grad=True params

    it = iter(data_loader) # iter is a python function that returns an iterator.
    # item = next(it) is the equivalent of doing for item in object_loader

    total, n = 0.0, 0
    step = 0
    losses = []
    
    # Create progress bar
    pbar = tqdm(total=steps, desc="Training", unit="step")
    
    while steps is None or step < steps:
        try:
            center, context, negative = next(it)
        except StopIteration:
            it = iter(data_loader)
            center, context, negative = next(it)

        center   = center.to(device, dtype=torch.long) # Tensors expect long (int64)
        context  = context.to(device, dtype=torch.long)
        negative = negative.to(device, dtype=torch.long)

        opt.zero_grad() # Initializes gradient to 0 so that the last step's gradients don't accumulate
        loss = model(center, context, negative) # calculates loss
        loss.backward() # computes gradients using backpropagation. This gradient gets stored in the model's parameters
        opt.step() # updates the model's parameters using the computed gradients

        total += loss.item()
        n += 1
        step += 1
        
        # Update progress bar
        pbar.update(1)
        
        if step % log_every == 0:
            avg_loss = total/n
            losses.append(avg_loss)
            plot_losses(losses, out_path="src/artifacts/training_loss.png")
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
            total, n = 0.0, 0
        
        # Evaluate nearest neighbors every 2000 steps
        if step % 2000 == 0 and itos is not None and log_file is not None:
            model.eval()
            with torch.no_grad():
                # Get current embeddings
                E = model.center_embeds.weight.cpu().numpy()
                norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
                E = E / norms
                stoi = {w: i for i, w in enumerate(itos)}
                
                # Log nearest neighbors
                with open(log_file, "a") as f:
                    f.write(f"\nStep {step}:\n")
                    for word in ["king", "paris", "computer"]:
                        neighbors = nearest(word, E, stoi, itos, k=5)
                        f.write(f"{word} → {neighbors}\n")
            model.train()
    
    pbar.close()
    return model, losses

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
    ds = SkipGramDataset(ids_path=ids_path, vocab_counts=counts, max_window=5, num_negatives=10)
    dl = DataLoader(ds, batch_size=1024, shuffle=True, num_workers=0)

    # Model
    model = SkipGramNS(vocab_size=V, embedding_dim=100)

    # Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_file = "src/artifacts/training_log.txt"
    
    # Initialize log file
    with open(log_file, "w") as f:
        f.write("Word2Vec Training Log\n")
        f.write("=" * 30 + "\n")
    
    model, losses = train_skipgram(model, dl, lr=0.002, steps=50000, device=device, 
                                   log_every=200, itos=itos, log_file=log_file)

    # Save vectors for inference
    save_vectors(model, itos)