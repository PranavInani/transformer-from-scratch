import numpy as np
from datasets import load_dataset
from config import get_config

def analyze_seq_len():
    config = get_config()
    ds_raw = load_dataset(f"{config['datasource']}")

    src_lengths = []
    tgt_lengths = []

    for split in ['train', 'validation', 'test']:
        for item in ds_raw[split]:
            src_len = len(item['translation'][config['lang_src']].split())
            tgt_len = len(item['translation'][config['lang_tgt']].split())
            src_lengths.append(src_len)
            tgt_lengths.append(tgt_len)

    print(f"Source language (max, min, avg): {max(src_lengths)}, {min(src_lengths)}, {sum(src_lengths)/len(src_lengths)}")
    print(f"Target language (max, min, avg): {max(tgt_lengths)}, {min(tgt_lengths)}, {sum(tgt_lengths)/len(tgt_lengths)}")
    print(f"Source language 99th percentile: {np.percentile(src_lengths, 99)}")
    print(f"Target language 99th percentile: {np.percentile(tgt_lengths, 99)}")

if __name__ == "__main__":
    analyze_seq_len()