import os
import json
import numpy as np

from bpe_tokenzier import train_bpe, BPETokenizer
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    return vocab, merges


@hydra.main(version_base="1.3", config_path="config", config_name="config.yaml")
def main(cfg: DictConfig):
    print(cfg)
    train_path = to_absolute_path(cfg.text_data.train_path)
    val_path = to_absolute_path(cfg.text_data.val_path)
    vocab_path = to_absolute_path(cfg.text_data.vocab_path)
    merges_path = to_absolute_path(cfg.text_data.merges_path)
    special_tokens = cfg.text_data.special_tokens
    vocab_size = cfg.text_data.vocab_size


    # extract file name of vocab_path and merges_path and then append vocab_size to the file name
    # Strip .json extension first (if present)
    vocab_base = os.path.splitext(os.path.basename(vocab_path))[0]
    merges_base = os.path.splitext(os.path.basename(merges_path))[0]

    # Add vocab size to filename
    vocab_file_name = f"{vocab_base}_{vocab_size}.json"
    merges_file_name = f"{merges_base}_{vocab_size}.json"

    # Rebuild full paths
    vocab_path = os.path.join(os.path.dirname(vocab_path), vocab_file_name)
    merges_path = os.path.join(os.path.dirname(merges_path), merges_file_name)

    vocab, merges = run_train_bpe(
        input_path=train_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    ## convert bytes to string
    vocab = {key: val.decode('utf-8', errors='ignore') for key, val in vocab.items()}
    merges = [(val[0].decode('utf-8', errors='ignore'), val[1].decode('utf-8', errors='ignore')) for val in merges]
    # save vocab and merges to json after converting bytes to string
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)
    with open(merges_path, "w") as f:
        json.dump(merges, f)

@hydra.main(version_base="1.3", config_path="config", config_name="config.yaml")
def encode_train_val_data(cfg: DictConfig):
    print(cfg)
    train_path = to_absolute_path(cfg.text_data.train_path)
    val_path = to_absolute_path(cfg.text_data.val_path)
    vocab_path = to_absolute_path(cfg.text_data.vocab_path)
    merges_path = to_absolute_path(cfg.text_data.merges_path)
    special_tokens = cfg.text_data.special_tokens
    vocab_size = cfg.text_data.vocab_size

    # extract file name of vocab_path and merges_path and then append vocab_size to the file name
    # Strip .json extension first (if present)
    vocab_base = os.path.splitext(os.path.basename(vocab_path))[0]
    merges_base = os.path.splitext(os.path.basename(merges_path))[0]

    # Add vocab size to filename
    vocab_file_name = f"{vocab_base}_{vocab_size}.json"
    merges_file_name = f"{merges_base}_{vocab_size}.json"

    # Rebuild full paths
    vocab_path = os.path.join(os.path.dirname(vocab_path), vocab_file_name)
    merges_path = os.path.join(os.path.dirname(merges_path), merges_file_name)

    
    bpe_tokenizer = BPETokenizer.from_files(vocab_path, merges_path, special_tokens)

    # Load train and val data as python str
    # with open(train_path, "r") as f:
    #     train_data = f.read()
    # with open(val_path, "r") as f:
    #     val_data = f.read()

    # Encode train and val data
    train_data_encoded = bpe_tokenizer.encode_big_text(train_path)
    val_data_encoded = bpe_tokenizer.encode_big_text(val_path)

    # Save train and val data encoded
    with open(train_path.replace(".txt", f"_{vocab_size}_encoded.npy"), "wb") as f:
        np.save(f, train_data_encoded)
    with open(val_path.replace(".txt", f"_{vocab_size}_encoded.npy"), "wb") as f:
        np.save(f, val_data_encoded)

@hydra.main(version_base="1.3", config_path="config", config_name="config.yaml")
def encode_val_data(cfg: DictConfig):
    print(cfg)
    train_path = to_absolute_path(cfg.text_data.train_path)
    val_path = to_absolute_path(cfg.text_data.val_path)
    vocab_path = to_absolute_path(cfg.text_data.vocab_path)
    merges_path = to_absolute_path(cfg.text_data.merges_path)
    special_tokens = cfg.text_data.special_tokens
    vocab_size = cfg.text_data.vocab_size

    # extract file name of vocab_path and merges_path and then append vocab_size to the file name
    # Strip .json extension first (if present)
    vocab_base = os.path.splitext(os.path.basename(vocab_path))[0]
    merges_base = os.path.splitext(os.path.basename(merges_path))[0]

    # Add vocab size to filename
    vocab_file_name = f"{vocab_base}_{vocab_size}.json"
    merges_file_name = f"{merges_base}_{vocab_size}.json"

    # Rebuild full paths
    vocab_path = os.path.join(os.path.dirname(vocab_path), vocab_file_name)
    merges_path = os.path.join(os.path.dirname(merges_path), merges_file_name)

    
    bpe_tokenizer = BPETokenizer.from_files(vocab_path, merges_path, special_tokens)

    # Load train and val data as python str
    # with open(train_path, "r") as f:
    #     train_data = f.read()
    # with open(val_path, "r") as f:
    #     val_data = f.read()

    # Encode train and val data
    print(f'Start encoding train data......')
    train_data_encoded = bpe_tokenizer.encode_big_text_by_chunk_size(train_path, chunk_size=1000)

    print(f'Start encoding val data......')
    val_data_encoded = bpe_tokenizer.encode_big_text_by_chunk_size(val_path, chunk_size=1000)

    # Save train and val data encoded
    print(f'Start saving train data......')
    with open(train_path.replace(".txt", f"_{vocab_size}_encoded.npy"), "wb") as f:
        np.save(f, train_data_encoded)
    print(f'Start saving val data......')
    with open(val_path.replace(".txt", f"_{vocab_size}_encoded_test.npy"), "wb") as f:
        np.save(f, val_data_encoded)

def decode_train_val_data(cfg: DictConfig):
    print(cfg)
    train_path = to_absolute_path(cfg.text_data.train_path)
    val_path = to_absolute_path(cfg.text_data.val_path)
    vocab_path = to_absolute_path(cfg.text_data.vocab_path)
    merges_path = to_absolute_path(cfg.text_data.merges_path)
    vocab_size = cfg.text_data.vocab_size
    

if __name__ == "__main__":
    # main()
    # encode_train_val_data()
    encode_val_data()

    # encoded_val_path = to_absolute_path("data/TinyStoriesV2-GPT4-valid_10000_encoded_test.npy")
    # encoded_val_data = np.load(encoded_val_path)
    # print(encoded_val_data)
