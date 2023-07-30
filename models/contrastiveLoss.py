import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, similarity, label):
        # Calculate contrastive loss
        loss_contrastive = torch.mean((1 - label) * torch.pow(similarity, 2) + 
                                      (label) * torch.pow(torch.clamp(self.margin - similarity, min=0.0), 2))
        return loss_contrastive
