# attention.py — MultiHeadAttention, scaled_dot_product_attention

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Compute scaled dot-product attention.
    Args:
        query: (batch_size, num_heads, seq_len_q, d_k)
        key: (batch_size, num_heads, seq_len_k, d_k)
        value: (batch_size, num_heads, seq_len_k, d_v)
        mask: (batch_size, 1, seq_len_q, seq_len_k) or (batch_size, seq_len_q, seq_len_k), True where to mask
    Returns:
        attention_output: (batch_size, num_heads, seq_len_q, d_v)
        attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    attention_weights = F.softmax(scores, dim=-1)
    attention_output = torch.matmul(attention_weights, value)
    return attention_output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0
        self.d_k = d_model // nhead
        self.nhead = nhead
        self.d_model = d_model

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_k, d_model)
            mask: (batch_size, seq_len_q, seq_len_k) or (batch_size, 1, 1, seq_len_k), True where to mask
        Returns:
            output: (batch_size, seq_len_q, d_model)
            attention_weights: (batch_size, nhead, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)

        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)  # (batch, nhead, seq_q, d_k)
        K = self.w_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

        # Adjust mask to broadcast over attention heads.
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_k)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch, 1, seq_q, seq_k)
            elif mask.dim() != 4:
                raise ValueError(f"Expected attention mask with 2, 3, or 4 dims, got {mask.dim()}")

            mask = mask.expand(-1, self.nhead, Q.size(2), -1)  # (batch, nhead, seq_q, seq_k)

        # Scaled dot-product attention
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(attn_output)

        return output, attn_weights
