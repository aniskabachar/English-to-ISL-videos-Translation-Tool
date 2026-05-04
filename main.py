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
    model = ISLTransformer(len(src_tokenizer.stoi), len(tgt_tokenizer.stoi), d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, max_len)
    model.to(device)
    return model

def train_model(model, device, num_epochs=epochs):
    # Data
    dataset = ISLDataset('data.csv', src_tokenizer, tgt_tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loss
    criterion = LabelSmoothingLoss(len(tgt_tokenizer.stoi))

    # Training
    for epoch in range(num_epochs):
        loss = train_epoch(model, loader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

    torch.save({
        'model_state_dict': model.state_dict(),
        'src_vocab_size': len(src_tokenizer.stoi),
        'tgt_vocab_size': len(tgt_tokenizer.stoi),
    }, checkpoint_path)
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
    parser.add_argument('--epochs', type=int, default=epochs)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(device)

    if args.mode in ['train', 'both']:
        train_model(model, device, args.epochs)
    elif os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            print(f'The checkpoint at {checkpoint_path} was saved with an older vocabulary/model shape.')
            print('Run `python main.py --mode train` once to rebuild it with the real data vocabulary.')
            return
        print(f'Loaded checkpoint: {checkpoint_path}')
    else:
        print(f'No checkpoint found at {checkpoint_path}.')
        print('Run `python main.py --mode train` once to train and save the model.')
        print('For a quick smoke test, run `python main.py --mode train --epochs 1`.')
        return

    if args.mode in ['infer', 'both']:
        run_inference(model, args.text, device)

if __name__ == '__main__':
    main()
