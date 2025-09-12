
import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchHardTripletLoss(nn.Module):
    # Batch-hard triplet loss from 'In Defense of the Triplet Loss for Person Re-Identification'
    # Assumes batch is organized as P classes * K samples each (PK sampling).
    def __init__(self, margin: float=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # Pairwise distance matrix
        dist = torch.cdist(embeddings, embeddings, p=2)
        N = embeddings.size(0)
        labels = labels.unsqueeze(1)
        mask_pos = (labels == labels.t()).float()
        mask_neg = (labels != labels.t()).float()

        # For each anchor, hardest positive and hardest negative
        dist_pos = (dist * mask_pos + (1 - mask_pos) * (-1e9)).max(1)[0]
        dist_neg = (dist * mask_neg + (1 - mask_neg) * (1e9)).min(1)[0]

        loss = F.relu(dist_pos - dist_neg + self.margin).mean()
        return loss
