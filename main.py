# main.py

import torch
import torch.optim as optim
import argparse
import os
from torch.utils.data import DataLoader
from model import ISLTransformer
from dataset import ISLDataset, src_tokenizer, tgt_tokenizer, collate_fn
from train import train_epoch, LabelSmoothingLoss
from inference import greedy_decode, beam_search
from config import *

def build_model(device):
    model = ISLTransformer(src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, max_len)
    model.to(device)
    return model

def train_model(model, device):
    # Data
    dataset = ISLDataset('data.csv', src_tokenizer, tgt_tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loss
    criterion = LabelSmoothingLoss(tgt_vocab_size)

    # Training
    for epoch in range(epochs):
        loss = train_epoch(model, loader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

    torch.save(model.state_dict(), checkpoint_path)
    print(f'Saved checkpoint: {checkpoint_path}')

def run_inference(model, src_text, device):
    src_indices = src_tokenizer.encode(src_text)
    src = torch.tensor([src_indices], device=device)
    src_mask = (src == src_tokenizer.stoi[PAD_TOKEN]).unsqueeze(1).unsqueeze(2).to(device)

    # Greedy
    greedy_output = greedy_decode(model, src, src_mask, device=device)
    greedy_gloss = tgt_tokenizer.decode(greedy_output)
    print(f'Greedy: {greedy_gloss}')

    # Beam
    beam_output = beam_search(model, src, src_mask, device=device)
    beam_gloss = tgt_tokenizer.decode(beam_output)
    print(f'Beam: {beam_gloss}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'infer', 'both'], default='infer')
    parser.add_argument('--text', default='have you seen my computer')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(device)

    if args.mode in ['train', 'both']:
        train_model(model, device)
    elif os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f'Loaded checkpoint: {checkpoint_path}')
    else:
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Run `python main.py --mode train` first."
        )

    if args.mode in ['infer', 'both']:
        run_inference(model, args.text, device)

if __name__ == '__main__':
    main()
