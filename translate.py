import torch
import torch.nn as nn
from pathlib import Path
from config import get_config, latest_weights_file_path 
from model import build_transformer
from tokenizers import Tokenizer
from datasets import load_dataset
from dataset import BilingualDataset, causal_mask
import argparse
import sys

def beam_search_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device, beam_size=3):
    """
    Performs beam search decoding for the transformer model.
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    
    # Initialize the beam with the start token
    sequences = [(torch.tensor([[sos_idx]], device=device), 0.0)]
    completed_sequences = []
    
    # Generate up to max_len tokens
    for _ in range(max_len):
        all_candidates = []
        
        # For each existing sequence
        for seq, score in sequences:
            # If the sequence is complete, add it to the completed list
            if seq[0, -1].item() == eos_idx:
                completed_sequences.append((seq, score))
                continue
                
            # Create the decoder mask for the current sequence
            decoder_mask = causal_mask(seq.size(1)).type_as(source_mask).to(device)
            
            # Get the decoder output for the current sequence
            out = model.decode(encoder_output, source_mask, seq, decoder_mask)
            
            # Get probabilities for the next token
            prob = model.project(out[:, -1])
            log_prob = nn.functional.log_softmax(prob, dim=-1)
            
            # Get the top k candidates
            topk_prob, topk_idx = torch.topk(log_prob, beam_size)
            
            # Create new candidate sequences
            for i in range(beam_size):
                next_token = topk_idx[0, i].unsqueeze(0).unsqueeze(0)
                next_score = score + topk_prob[0, i].item()
                candidate = (torch.cat([seq, next_token], dim=1), next_score)
                all_candidates.append(candidate)
        
        # Select top-k candidates
        all_candidates.sort(key=lambda x: x[1] / len(x[0].squeeze(0)), reverse=True)  # Normalize by length
        sequences = all_candidates[:beam_size]
        
        # If all sequences are completed, break
        if all(seq[0, -1].item() == eos_idx for seq, _ in sequences):
            break
    
    # Add any remaining sequences to completed ones
    completed_sequences.extend([s for s in sequences if s[0][0, -1].item() == eos_idx])
    
    # If no sequence ended with EOS token, just use what we have
    if not completed_sequences:
        completed_sequences = sequences
    
    # Return the sequence with highest normalized score
    completed_sequences.sort(key=lambda x: x[1] / len(x[0].squeeze(0)), reverse=True)
    return completed_sequences[0][0]

def translate(sentence, beam_size=3):
    """
    Translates a sentence from the source language to the target language.
    """
    # Define the device, tokenizers, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    config = get_config()
    
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
    
    model = build_transformer(
        tokenizer_src.get_vocab_size(), 
        tokenizer_tgt.get_vocab_size(), 
        config["seq_len"], 
        config['seq_len'], 
        d_model=config['d_model']
    ).to(device)
    
    # Load the pretrained weights
    model_filename = latest_weights_file_path(config)
    if model_filename is None:
        print("No trained model found. Please train the model first.")
        return None
        
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    
    # If the sentence is a number use it as an index to the test set
    label = ""
    if type(sentence) == int or sentence.isdigit():
        id = int(sentence)
        ds = load_dataset(f"{config['datasource']}", split='test')
        sentence = ds[id]['translation'][config['lang_src']]
        label = ds[id]["translation"][config['lang_tgt']]
    
    seq_len = config['seq_len']

    # Translate the sentence
    model.eval()
    with torch.no_grad():
        # Precompute the encoder output and reuse it for every generation step
        source = tokenizer_src.encode(sentence)
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)
        ], dim=0).to(device)
        source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)

        # Print the source sentence and target start prompt
        if label != "": print(f"{f'ID: ':>12}{id}") 
        print(f"{f'SOURCE: ':>12}{sentence}")
        if label != "": print(f"{f'TARGET: ':>12}{label}") 
        print(f"{f'PREDICTED: ':>12}", end='')

        # Modify this part to use beam search
        if beam_size > 1:
            decoder_output = beam_search_decode(model, source.unsqueeze(0), source_mask, tokenizer_src, tokenizer_tgt, seq_len, device, beam_size)
            # Print the beam search result
            translated = tokenizer_tgt.decode(decoder_output.squeeze().tolist())
            print(translated)
        else:
            # Use the original greedy decoding
            encoder_output = model.encode(source.unsqueeze(0), source_mask)
            decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)
            
            # Generate the translation token by token
            while decoder_input.size(1) < seq_len:
                # build mask for target and calculate output
                decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)
                out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

                # project next token
                prob = model.project(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

                # print the translated token
                token = tokenizer_tgt.decode([next_word.item()])
                print(f"{token}", end='')

                # break if we predict the end of sentence token
                if next_word == tokenizer_tgt.token_to_id('[EOS]'):
                    break
            translated = tokenizer_tgt.decode(decoder_input.squeeze().tolist())
    
    return translated

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate text using trained Transformer model')
    parser.add_argument('text', type=str, help='Text to translate (use number for test set index)')
    parser.add_argument('--beam_size', type=int, default=3, help='Beam size for beam search decoding')
    parser.add_argument('--model', type=str, default='latest', help='Model checkpoint to use')
    args = parser.parse_args()
    
    config = get_config()
    config['preload'] = args.model
    
    translated = translate(args.text, beam_size=args.beam_size)
    print("\nFinal translation:", translated)

