import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

class NegativeSampler:
    def __init__(self, counts, power=0.75, seed=0):
        V = max(counts.keys()) + 1
        freqs = np.zeros(V, dtype=np.float64)
        for wid, c in counts.items():
            freqs[wid] = c
        probs = np.power(freqs, power)
        probs /= probs.sum()
        self.probs = probs
        self.V = V
        self.rng = np.random.default_rng(seed)

    def draw(self, batch_size, K):
        return self.rng.choice(self.V, size=(batch_size, K), p=self.probs)


class SkipGramDataset(Dataset):
    def __init__(self, ids_path="src/artifacts/text8_ids.npy", vocab_counts=None, 
                 max_window=5, num_negatives=5):
        self.ids = np.load(ids_path).astype(np.int64)
        self.max_window = max_window
        self.num_negatives = num_negatives
        self.neg_sampler = NegativeSampler(vocab_counts)
        self.pairs = self._generate_pairs()

    def _generate_pairs(self):
        pairs = []
        n = len(self.ids)
        for i, center in enumerate(self.ids):
            w = random.randint(1, self.max_window)
            left, right = max(0, i-w), min(n, i+w+1)
            for j in range(left, right):
                if j == i: 
                    continue
                pairs.append((center, self.ids[j]))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        negatives = self.neg_sampler.draw(1, self.num_negatives)[0]
        return (torch.tensor(center, dtype=torch.long),
                torch.tensor(context, dtype=torch.long),
                torch.tensor(negatives, dtype=torch.long))