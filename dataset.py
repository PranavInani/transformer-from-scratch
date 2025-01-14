import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)


    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        src = self.tokenizer_src.encode(item['translation'][self.src_lang]).ids
        tgt = self.tokenizer_tgt.encode(item['translation'][self.tgt_lang]).ids

        # Truncate if necessary
        if len(src) > self.seq_len - 2:
            src = src[:self.seq_len - 2]
        if len(tgt) > self.seq_len - 1:
            tgt = tgt[:self.seq_len - 1]

        # add sos and eos tokens
        enc_num_padding_tokens = self.seq_len - len(src) - 2
        dec_num_padding_tokens = self.seq_len - len(tgt) - 1

        # add both sos and eos
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(src, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ], 
            dim=0
        )
        # add only sos token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(tgt, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ], 
            dim=0
        )

        # add only eos token
        label = torch.cat(
            [
                torch.tensor(tgt, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ], 
            dim=0
        )

        # double check the size of the tensors to make sure they are all seq_len
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            'label': label,
            'src_text' : item['translation'][self.src_lang],
            'tgt_text' : item['translation'][self.tgt_lang]
        }

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

