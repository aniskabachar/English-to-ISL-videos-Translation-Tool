# inference.py — greedy_decode(model, src), beam_search(model, src, k=5)

import torch
import torch.nn.functional as F
from config import *
from dataset import src_tokenizer, tgt_tokenizer

def greedy_decode(model, src, src_mask, max_len=max_len, device='cpu'):
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        tgt = torch.tensor([[tgt_tokenizer.stoi[BOS_TOKEN]]], device=device)
        for _ in range(max_len):
            tgt_len = tgt.size(1)
            tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().unsqueeze(0).to(device)
            # Memory mask: (1, tgt_len, src_len)
            src_len = src.size(1)
            src_pad = (src == src_tokenizer.stoi[PAD_TOKEN]).squeeze(0)  # (src_len,)
            memory_mask = src_pad.unsqueeze(0).expand(tgt_len, -1).unsqueeze(0).to(device)
            output = model.decode(tgt, memory, tgt_mask, memory_mask)
            logits = model.output_layer(output[:, -1, :])
            next_token = logits.argmax(dim=-1).item()
            if next_token == tgt_tokenizer.stoi[EOS_TOKEN]:
                break
            tgt = torch.cat([tgt, torch.tensor([[next_token]], device=device)], dim=1)
        return tgt.squeeze(0).tolist()

def beam_search(model, src, src_mask, k=beam_size, max_len=max_len, device='cpu'):
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        src_len = src.size(1)
        src_pad = (src == src_tokenizer.stoi[PAD_TOKEN]).squeeze(0)  # (src_len,)
        # Start with BOS
        beams = [([tgt_tokenizer.stoi[BOS_TOKEN]], 0.0)]  # (sequence, score)
        completed = []

        for _ in range(max_len):
            new_beams = []
            for seq, score in beams:
                if seq[-1] == tgt_tokenizer.stoi[EOS_TOKEN]:
                    completed.append((seq, score))
                    continue
                tgt = torch.tensor([seq], device=device)
                tgt_len = len(seq)
                tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().unsqueeze(0).to(device)
                memory_mask = src_pad.unsqueeze(0).expand(tgt_len, -1).unsqueeze(0).to(device)
                output = model.decode(tgt, memory, tgt_mask, memory_mask)
                logits = model.output_layer(output[:, -1, :])
                probs = F.log_softmax(logits, dim=-1).squeeze(0)
                top_k_probs, top_k_indices = probs.topk(k)
                for prob, idx in zip(top_k_probs, top_k_indices):
                    new_seq = seq + [idx.item()]
                    new_score = score + prob.item()
                    new_beams.append((new_seq, new_score))
            # Select top k
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:k]
            if all(seq[-1] == tgt_tokenizer.stoi[EOS_TOKEN] for seq, _ in beams):
                break
        # Add remaining to completed
        completed.extend(beams)
        # Return the best
        best_seq, _ = max(completed, key=lambda x: x[1])
        return best_seq
