import numpy as np
from datasets import load_dataset
from config import get_config
from tokenizers import Tokenizer
from pathlib import Path

def analyze_seq_len():
    config = get_config()
    ds_raw = load_dataset(f"{config['datasource']}")
    
    # Load the tokenizers
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
    
    src_lengths = []
    tgt_lengths = []

    for split in ['train', 'validation', 'test']:
        for item in ds_raw[split]:
            # Get token counts using the BPE tokenizer
            src_tokens = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
            tgt_tokens = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
            
            src_len = len(src_tokens)
            tgt_len = len(tgt_tokens)
            
            src_lengths.append(src_len)
            tgt_lengths.append(tgt_len)

    print(f"BPE Source tokens (max, min, avg): {max(src_lengths)}, {min(src_lengths)}, {sum(src_lengths)/len(src_lengths)}")
    print(f"BPE Target tokens (max, min, avg): {max(tgt_lengths)}, {min(tgt_lengths)}, {sum(tgt_lengths)/len(tgt_lengths)}")
    print(f"BPE Source tokens 99th percentile: {np.percentile(src_lengths, 99)}")
    print(f"BPE Target tokens 99th percentile: {np.percentile(tgt_lengths, 99)}")
    
    # Additional useful percentiles
    print(f"BPE Source tokens 95th percentile: {np.percentile(src_lengths, 95)}")
    print(f"BPE Target tokens 95th percentile: {np.percentile(tgt_lengths, 95)}")

if __name__ == "__main__":
    analyze_seq_len()