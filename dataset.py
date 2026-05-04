# dataset.py — ISLDataset(Dataset), Tokenizer, collate_fn with padding + masks

import torch
from torch.utils.data import Dataset
import pandas as pd
import string
import os
from config import *

class Tokenizer:
    def __init__(self, vocab, special_tokens):
        self.vocab = vocab
        self.special_tokens = special_tokens
        self.itos = {i: token for token, i in vocab.items()}
        self.stoi = vocab

    def encode(self, text, add_bos_eos=True):
        tokens = text.split()
        if add_bos_eos:
            tokens = [BOS_TOKEN] + tokens + [EOS_TOKEN]
        indices = [self.stoi.get(token, self.stoi[UNK_TOKEN]) for token in tokens]
        return indices

    def decode(self, indices):
        tokens = [self.itos.get(i, UNK_TOKEN) for i in indices]
        return ' '.join(tokens)

def preprocess_src(text):
    return ''.join(c for c in str(text).lower() if c not in string.punctuation).strip()

def preprocess_tgt(text):
    return str(text).upper().strip()

def build_vocab(tokens):
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, BOS_TOKEN: 2, EOS_TOKEN: 3}
    for token in sorted(set(tokens)):
        if token and token not in vocab:
            vocab[token] = len(vocab)
    return vocab

def build_vocabs_from_csv(data_path='data.csv'):
    if not os.path.exists(data_path):
        return build_vocab([]), build_vocab([])

    data = pd.read_csv(data_path)
    src_tokens = []
    tgt_tokens = []

    for sentence in data['Sentence']:
        src_tokens.extend(preprocess_src(sentence).split())

    for glosses in data['SIGN GLOSSES']:
        tgt_tokens.extend(preprocess_tgt(glosses).split())

    return build_vocab(src_tokens), build_vocab(tgt_tokens)

src_vocab, tgt_vocab = build_vocabs_from_csv()

src_tokenizer = Tokenizer(src_vocab, [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN])
tgt_tokenizer = Tokenizer(tgt_vocab, [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN])

class ISLDataset(Dataset):
    def __init__(self, data_path, src_tokenizer, tgt_tokenizer, max_len=max_len):
        self.data = pd.read_csv(data_path)  # CSV with 'Sentence', 'SIGN GLOSSES'
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        src_text = preprocess_src(row['Sentence'])
        tgt_text = preprocess_tgt(row['SIGN GLOSSES'])

        src_indices = self.src_tokenizer.encode(src_text)
        tgt_indices = self.tgt_tokenizer.encode(tgt_text)

        # Pad to max_len
        src_indices = src_indices[:self.max_len] + [self.src_tokenizer.stoi[PAD_TOKEN]] * (self.max_len - len(src_indices))
        tgt_indices = tgt_indices[:self.max_len] + [self.tgt_tokenizer.stoi[PAD_TOKEN]] * (self.max_len - len(tgt_indices))

        return torch.tensor(src_indices), torch.tensor(tgt_indices)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)

    pad_id = src_tokenizer.stoi[PAD_TOKEN]

    # Src mask: (batch, 1, 1, seq) True where pad
    src_mask = (src_batch == pad_id).unsqueeze(1).unsqueeze(2)

    # Tgt mask: (batch, seq, seq) True where to mask (causal or pad)
    batch_size, tgt_len = tgt_batch.size()
    causal_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
    pad_positions = (tgt_batch == pad_id)  # (batch, seq)
    tgt_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1) | pad_positions.unsqueeze(1).expand(-1, tgt_len, -1)

    # Memory mask: (batch, tgt_len, src_len) True where src pad
    src_len = src_batch.size(1)
    memory_mask = (src_batch == pad_id).unsqueeze(1).expand(-1, tgt_len, -1)

    return src_batch, tgt_batch, src_mask, tgt_mask, memory_mask
