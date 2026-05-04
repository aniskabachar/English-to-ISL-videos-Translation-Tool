# train.py — train_epoch(model, loader), LabelSmoothingLoss

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, smoothing=label_smoothing):
        super(LabelSmoothingLoss, self).__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        """
        pred: (batch, seq_len, vocab_size)
        target: (batch, seq_len)
        """
        pred = pred.reshape(-1, self.vocab_size)
        target = target.reshape(-1)
        true_dist = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist += self.smoothing / self.vocab_size
        # Mask out padding
        mask = (target != 0).float()  # Assuming 0 is PAD
        loss = torch.sum(-true_dist * F.log_softmax(pred, dim=1), dim=1) * mask
        return loss.sum() / mask.sum()

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt, src_mask, tgt_mask, memory_mask in loader:
        src, tgt = src.to(device), tgt.to(device)
        src_mask, tgt_mask, memory_mask = src_mask.to(device), tgt_mask.to(device), memory_mask.to(device)

        optimizer.zero_grad()
        output = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :-1, :-1], memory_mask[:, :-1, :])  # tgt input without last token
        loss = criterion(output, tgt[:, 1:])  # target without first token
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
