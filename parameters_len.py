import torch
from model import build_transformer
from config import get_config
import json
from tokenizers import Tokenizer
from pathlib import Path

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_tokenizer_size(tokenizer_path):
    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()
        return vocab_size
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None

if __name__ == "__main__":
    config = get_config()
    
    # Get tokenizer paths
    src_tokenizer_path = str(Path(config['tokenizer_file'].format(config['lang_src'])))
    tgt_tokenizer_path = str(Path(config['tokenizer_file'].format(config['lang_tgt'])))
    
    # Get vocab sizes
    src_vocab_size = get_tokenizer_size(src_tokenizer_path)
    tgt_vocab_size = get_tokenizer_size(tgt_tokenizer_path)
    
    if not src_vocab_size or not tgt_vocab_size:
        print("Failed to get vocabulary sizes. Please ensure tokenizer files exist.")
        exit(1)

    model = build_transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_seq_len=config['seq_len'],
        tgt_seq_len=config['seq_len'],
        d_model=config['d_model']
    )
    
    total_params = count_parameters(model)
    print(f"Total number of trainable parameters: {total_params:,}")
    print(f"Source vocabulary size: {src_vocab_size}")
    print(f"Target vocabulary size: {tgt_vocab_size}")
    
    # Print parameter breakdown by component
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel():,}")