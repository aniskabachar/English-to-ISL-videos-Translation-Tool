# config.py — all hyperparams

# Model hyperparameters
d_model = 512
nhead = 8
num_encoder_layers = 4
num_decoder_layers = 4
dim_feedforward = 2048  # standard for transformers
dropout = 0.1

# Vocabulary sizes
src_vocab_size = 10000  # English vocab
tgt_vocab_size = 2000   # ISL gloss vocab

# Sequence length
max_len = 128

# Special tokens
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'

# Training hyperparameters
batch_size = 32
epochs = 10
lr = 1e-4
warmup_steps = 4000
label_smoothing = 0.1

# Inference
beam_size = 5

# Data
data_split = [0.8, 0.1, 0.1]  # train/val/test
checkpoint_path = 'isl_transformer.pt'
