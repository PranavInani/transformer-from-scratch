# Transformer from Scratch

This project implements a transformer model for bilingual translation from English to Hindi using subword tokenization (BPE). The implementation includes training, evaluation, and inference scripts.

## Project Structure

- `analyze_seq_len.py`: Analyzes the sequence lengths of the dataset.
- `config.py`: Configuration file for the project.
- `dataset.py`: Defines the `BilingualDataset` class for handling the dataset.
- `inspect_ds.py`: Inspects the dataset structure and prints a sample.
- `model.py`: Defines the transformer model and its components.
- `parameters_len.py`: Counts the number of parameters in the model.
- `subword-level-translation-train.ipynb`: Jupyter notebook for training the model.
- `test.py`: Script for evaluating the model on the test dataset.
- `train.py`: Script for training the transformer model.
- `translate.py`: Script for translating sentences using the trained model.
- `.gitignore`: Specifies files and directories to be ignored by git.

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/PranavInani/transformer-from-scratch
    cd transformer-from-scratch
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Download and prepare the dataset:
    ```sh
    python inspect_ds.py
    ```

## Training

To train the model, run:
```sh
python train.py
```

Alternatively, you can use the Jupyter notebook `subword-level-translation-train.ipynb` to train the model interactively.

## Evaluation

To evaluate the model on the test dataset, run:
```sh
python test.py
```

## Inference

To translate a sentence using the trained model, run:
```sh
python translate.py "Your sentence here"
```

## Configuration

The configuration settings are defined in `config.py`. You can modify the settings as needed.

## Tokenization

The project uses Byte Pair Encoding (BPE) for subword tokenization. The tokenizers are trained and saved automatically if they do not exist.

## Model

The transformer model is defined in `model.py` and includes the following components:
- Input Embeddings
- Positional Encoding
- Multi-Head Attention
- Feed Forward Network
- Encoder and Decoder Blocks
- Projection Layer

## License

This project is licensed under the MIT License.

## Acknowledgements

This project is inspired by the original transformer paper "Attention is All You Need" by Vaswani et al.

