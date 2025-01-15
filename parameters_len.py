import torch
from model import build_transformer
from config import get_config
import json

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_vocab_size(tokenizer_path):
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    return len(tokenizer_data['model']['vocab'])

if __name__ == "__main__":
    config = get_config()
    
    # Get vocabulary sizes
    src_vocab_size = get_vocab_size('/home/pranav/nlp/transformer-from-scratch/tokenizer_hi.json')
    tgt_vocab_size = get_vocab_size('/home/pranav/nlp/transformer-from-scratch/tokenizer_en.json')

    model = build_transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_seq_len=config['seq_len'],
        tgt_seq_len=config['seq_len'],
        d_model=config['d_model']
    )
    total_params = count_parameters(model)
    print(f"Total number of trainable parameters: {total_params}")
    print(f"Source vocabulary size: {src_vocab_size}")
    print(f"Target vocabulary size: {tgt_vocab_size}")