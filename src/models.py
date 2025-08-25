import torch
import torch.nn as nn
import torch.nn.functional as F

def sgns_loss(pos_score, neg_score):
    return -(F.logsigmoid(pos_score).mean() + F.logsigmoid(-neg_score).sum(dim=1).mean())

class SkipGramNS(nn.Module): # Skip-gram negative sampling (sgns)
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.center_embeds = nn.Embedding(vocab_size, embedding_dim) # Initialized matrix with random values
        self.context_embeds = nn.Embedding(vocab_size, embedding_dim)

        # Initialization that matches the paper
        nn.init.uniform_(self.center_embeds.weight, -0.5/embedding_dim, 0.5/embedding_dim)
        nn.init.constant_(self.context_embeds.weight, 0.0)

    def forward(self, center_id, context_ids, negative_ids):
        v_center = self.center_embeds(center_id)
        v_pos = self.context_embeds(context_ids)
        v_neg = self.context_embeds(negative_ids)

        # Positive scores: dot per example and sum
        pos_score = (v_center * v_pos).sum(dim=1)

        # Negative scores: dot with each of the K negatives and sum
        neg_score = (v_center.unsqueeze(1) * v_neg).sum(dim=2)

        # Compute loss
        loss = sgns_loss(pos_score, neg_score)
        return loss
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
