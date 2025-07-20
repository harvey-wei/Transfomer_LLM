from bytephase import Tokenizer
import os
import regex as re
from typing import BinaryIO
from typing import Iterable, Iterator
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import Tuple
import heapq
import time
import json
from tqdm import tqdm


def _boundary_worker(args: tuple[str, int, int, bytes, int]) -> int:
    """Worker function for finding chunk boundaries
    the split_special_token is a bytestring - b'<|endoftext|>' which is used to separate documents.
    We do not want to merge tokens across documents, which is different from the single document BPE
    """
    file_path, initial_position, mini_chunk_size, doc_sep_token_bytes, file_size = args
    adjusted_position = initial_position

    with open(file_path, "rb") as f:
        f.seek(initial_position)
        while True:
            mini_chunk = f.read(mini_chunk_size)

            if mini_chunk == b"":  # EOF
                adjusted_position = file_size
                break

            found_at = mini_chunk.find(doc_sep_token_bytes)
            if found_at != -1:
                adjusted_position = initial_position + found_at
                break

            initial_position += mini_chunk_size

    return adjusted_position

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    doc_sep_token: bytes,
    mini_chunk_size: int = 4096
) -> list[int]:
    assert isinstance(doc_sep_token, bytes), "Special token must be bytes."

    if not hasattr(file, "name"):
        raise ValueError("File object must have a 'name' attribute (i.e., be an on-disk file).")

    # File size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    file_path = file.name
    args_list = [
        (file_path, chunk_boundaries[i], mini_chunk_size, doc_sep_token, file_size)
        for i in range(1, len(chunk_boundaries) - 1)
    ]

    with Pool(cpu_count()) as pool:
        adjusted = pool.map(_boundary_worker, args_list)

    assert len(adjusted) == len(chunk_boundaries) - 2

    for i, pos in enumerate(adjusted):
        chunk_boundaries[i + 1] = pos

    return sorted(set(chunk_boundaries))

def big_text_iterable(input_path: str) -> Iterable[str]:
    """
    Encode the text in memory efficient way
    """
    desired_num_chunks = 100000
    doc_sep_token_bytes = b"<|endoftext|>"
    

    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, desired_num_chunks, doc_sep_token_bytes)
    
    # get chunk_freq_table using multiple processes
    args_list = [(input_path, start_pos, end_pos, self.special_tokens) for start_pos, end_pos in zip(chunk_boundaries[:-1], chunk_boundaries[1:])]

    for input_path, start_pos, end_pos, special_tokens in args_list:
        with open(input_path, "rb") as f:
            f.seek(start_pos)
            text = f.read(end_pos - start_pos).decode("utf-8", errors="ignore")
        
        yield text

def encode_big_text(input_path: str) -> list[int]:
    """
    Encode a big text file
    """
    big_text_iterable = big_text_iterable(input_path)
    token_ids = []

    for text in big_text_iterable:
        print(f'Start encoding chunk ......')
        token_ids_chunk = encode(text)
        token_ids.extend(token_ids_chunk)
        print(f'One chunk tokenization done')

    return token_ids



tokenizer = Tokenizer()

vocab_size = 10000
tokenizer.train(file_path='data/TinyStoriesV2-GPT4-train.txt', vocab_size=vocab_size)

tokenizer.save("TinyStoriesV2-GPT4-train_10000_tokenzier_bytephase")

encoded = tokenizer.encode("Hello, world!")
print(f'encoded {encoded}')

decoded = tokenizer.decode(encoded)
print(f'decoded')


