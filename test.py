import torch
import torchmetrics
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

from config import get_config, latest_weights_file_path
from model import build_transformer
from dataset import BilingualDataset, causal_mask
from train import greedy_decode
from tokenizers import Tokenizer

def test_model():
    # Get configurations
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizers (now BPE)
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
    
    # Load dataset
    ds_raw = load_dataset(f"{config['datasource']}")
    test_ds = BilingualDataset(
        ds_raw['test'], 
        tokenizer_src, 
        tokenizer_tgt,
        config['lang_src'], 
        config['lang_tgt'], 
        config['seq_len']
    )
    test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # Load model
    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config['seq_len'],
        config['seq_len'],
        d_model=config['d_model']
    ).to(device)
    
    # Load pretrained weights
    model_filename = latest_weights_file_path(config)
    if model_filename is None:
        print("No trained model found. Please train the model first.")
        return
        
    print(f"Loading model from {model_filename}")
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    
    # Prepare metrics
    cer_metric = torchmetrics.CharErrorRate().to(device)
    wer_metric = torchmetrics.WordErrorRate().to(device)
    bleu_metric = torchmetrics.BLEUScore().to(device)
    
    # Track results
    source_texts = []
    expected_texts = []
    predicted_texts = []
    
    # Store sequence lengths for analysis
    src_seq_lengths = []
    tgt_seq_lengths = []
    
    # Evaluate on test set
    print("Starting evaluation on test dataset...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            
            # Record sequence lengths for analysis
            src_text = batch["src_text"][0]
            tgt_text = batch["tgt_text"][0]
            src_seq_lengths.append(len(src_text.split()))
            tgt_seq_lengths.append(len(tgt_text.split()))
            
            # Generate translation
            model_output = greedy_decode(
                model, 
                encoder_input, 
                encoder_mask, 
                tokenizer_src, 
                tokenizer_tgt, 
                config['seq_len'], 
                device
            )
            
            predicted_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())
            
            # Store results
            source_texts.append(src_text)
            expected_texts.append(tgt_text)
            predicted_texts.append(predicted_text)
    
    # Calculate metrics
    cer = cer_metric(predicted_texts, expected_texts)
    wer = wer_metric(predicted_texts, expected_texts)
    bleu = bleu_metric(predicted_texts, expected_texts)
    
    # Print metrics
    print("\nTest Results:")
    print(f"Character Error Rate (CER): {cer:.4f}")
    print(f"Word Error Rate (WER): {wer:.4f}")
    print(f"BLEU Score: {bleu:.4f}")
    
    # Find best and worst examples based on character error rate
    individual_cer = [torchmetrics.functional.char_error_rate(pred, exp) 
                      for pred, exp in zip(predicted_texts, expected_texts)]
    best_idx = np.argmin(individual_cer)
    worst_idx = np.argmax(individual_cer)
    
    print("\nBest Translation Example:")
    print(f"Source: {source_texts[best_idx]}")
    print(f"Target: {expected_texts[best_idx]}")
    print(f"Predicted: {predicted_texts[best_idx]}")
    print(f"CER: {individual_cer[best_idx]:.4f}")
    
    print("\nWorst Translation Example:")
    print(f"Source: {source_texts[worst_idx]}")
    print(f"Target: {expected_texts[worst_idx]}")
    print(f"Predicted: {predicted_texts[worst_idx]}")
    print(f"CER: {individual_cer[worst_idx]:.4f}")
    
    # Analysis of sequence lengths vs performance
    plt.figure(figsize=(10, 6))
    plt.scatter(src_seq_lengths, individual_cer, alpha=0.5)
    plt.xlabel('Source Sentence Length (words)')
    plt.ylabel('Character Error Rate')
    plt.title('Impact of Source Sentence Length on Translation Quality')
    plt.savefig('length_vs_error.png')
    print("\nAnalysis plot saved as 'length_vs_error.png'")
    
    # Save results to file
    with open('test_results.txt', 'w', encoding='utf-8') as f:
        f.write(f"Test Results:\n")
        f.write(f"Character Error Rate (CER): {cer:.4f}\n")
        f.write(f"Word Error Rate (WER): {wer:.4f}\n")
        f.write(f"BLEU Score: {bleu:.4f}\n")
        
        # Write some examples
        f.write("\nSample Translations:\n")
        for i in range(min(10, len(source_texts))):
            f.write(f"\nExample {i+1}:\n")
            f.write(f"Source: {source_texts[i]}\n")
            f.write(f"Target: {expected_texts[i]}\n")
            f.write(f"Predicted: {predicted_texts[i]}\n")
            f.write(f"CER: {individual_cer[i]:.4f}\n")
    
    print("Detailed results saved to 'test_results.txt'")

if __name__ == "__main__":
    test_model()