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
import numpy as np


def find_chunk_boundaries_single_process(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


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


def remove_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """Split on the special tokens like <|endoftext|> avoid tokenize across documents"""
    # list of tokens without special tokens
    segments = re.split("|".join(re.escape(tok) for tok in special_tokens), text)

    return segments

def pretokenize_by_chunk_boundary(args: tuple[str, int, int, list[str]]) -> defaultdict[tuple[bytes], int]:
    """Pretokenize the chunk of a large file
    The chunk is specified by byte index range [start_pos, end_pos) of the file.
    """
    file_path, start_pos, end_pos, special_tokens = args
    # Map bytes to int
    freq_table = defaultdict(int)
    with open(file_path, "rb") as f:
        f.seek(start_pos)
        text = f.read(end_pos - start_pos).decode("utf-8", errors="ignore")

        segments = remove_special_tokens(text, special_tokens)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for part in segments:
            for match in re.finditer(PAT, part):
                string = match.group()
                # Here the tuple of bytes integer value acts as key---pre-token is a tuple of bytes
                key = tuple(bytes([integer]) for integer in list(string.encode("utf-8")))
                freq_table[key] += 1

    return freq_table

def pretokenize_by_chunk_boundary_multi_process(input_path: str, desired_num_chunks: int, special_tokens: list[str], doc_sep_token_bytes: bytes = b"<|endoftext|>" ) -> list[bytes]:
    """Pretokenize the text file using multiple processes"""
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, desired_num_chunks, doc_sep_token_bytes)
    
    # get chunk_freq_table using multiple processes
    args_list = [(input_path, start_pos, end_pos, special_tokens) for start_pos, end_pos in zip(chunk_boundaries[:-1], chunk_boundaries[1:])]
    
    with Pool(cpu_count()) as pool:
        chunk_freq_table_list = pool.map(pretokenize_by_chunk_boundary, args_list)

    freq_table = defaultdict(int)
    for chunk_freq_table in chunk_freq_table_list:
        for key, value in chunk_freq_table.items():
            freq_table[key] += value

    return freq_table

def pretokenize_chunk(args: tuple[str, list[str]]) -> defaultdict[tuple[bytes], int]:
    """Seperating text into pretokens"""
    text, special_tokens = args
    segments = remove_special_tokens(text, special_tokens)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # Map bytes to int
    freq_table = defaultdict(int)
    for part in segments:
        for match in re.finditer(PAT, part):
            string = match.group()
            # Here the tuple of bytes integer value acts as key---pre-token is a tuple of bytes
            key = tuple(bytes([integer]) for integer in list(string.encode("utf-8")))
            freq_table[key] += 1

    return freq_table

def pretokenize_multi_process(input_path: str, desired_num_chunks: int, special_tokens: list[str], doc_sep_token_bytes: bytes = b"<|endoftext|>" ) -> list[bytes]:
    """Pretokenize the text file using multiple processes"""
    chunk_list = get_chunk_text_list_multi_process(input_path, desired_num_chunks, doc_sep_token_bytes)
    
    # get chunk_freq_table using multiple processes
    args_list = [(chunk, special_tokens) for chunk in chunk_list]
    
    with Pool(cpu_count()) as pool:
        chunk_freq_table_list = pool.map(pretokenize_chunk, args_list)

    freq_table = defaultdict(int)
    for chunk_freq_table in chunk_freq_table_list:
        for key, value in chunk_freq_table.items():
            freq_table[key] += value

    return freq_table

def get_chunk_text_worker(args: tuple[str, int, int]) -> str:
    file_path, start_pos, end_pos = args
    with open(file_path, "rb") as f:
        f.seek(start_pos)
        chunk = f.read(end_pos - start_pos).decode("utf-8", errors="ignore")
    return chunk

def get_chunk_text_list_multi_process(input_path: str, desired_num_chunks: int, doc_sep_token_bytes: bytes) -> list[str]:
    """Get the chunk text list using multiple processes"""
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, desired_num_chunks, doc_sep_token_bytes)

    args_list = [(input_path, start, end) for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:])]
    with Pool(cpu_count()) as pool:
        chunk_list = pool.map(get_chunk_text_worker, args_list)

    return chunk_list

def get_chunk_text_list_single_process(input_path: str, desired_num_chunks: int, special_tokens: list[str]) -> list[str]:
    """Get the chunk text list using single process"""
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries_single_process(f, desired_num_chunks, special_tokens)
    args_list = [(input_path, start, end) for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:])]
    chunk_list = [get_chunk_text_worker(args) for args in args_list]
    return chunk_list

def train_bpe_heap_optimized(pretoken_freq: dict[Tuple[bytes], int], vocab_size: int, special_tokens: list[str]):
    '''
    This function is optimized for heap operation
    Caveat: Vocab size is vocab_size - 1, because we need to reserve one slot for the unknown token.
    '''
    vocab = {}
    curr_id = 0
    for token in special_tokens:
        vocab[curr_id] = token.encode("utf-8")
        curr_id += 1
    for i in range(256):
        vocab[curr_id] = bytes([i])
        curr_id += 1

    merges = []
    pair2freq = defaultdict(int)

    # Initialize pair frequencies
    for token_seq, freq in pretoken_freq.items():
        for i in range(len(token_seq) - 1):
            pair = (token_seq[i], token_seq[i + 1])
            pair2freq[pair] += freq

    heap = [(-freq, pair) for pair, freq in pair2freq.items()]
    heapq.heapify(heap)

    # Note that pair2freq must be updated after each merge
    # While heap does not support pop non-min node, we need to check if the pair is in pair2freq

    while len(vocab) < vocab_size and heap:
        # Pop the most frequent valid pair
        while heap:
            # This might the old pair, but we will check if it is in pair2freq
            neg_freq, pair = heapq.heappop(heap)
            if pair in pair2freq and -neg_freq == pair2freq[pair]:
                # This is the valid pair, we can break out of the while loop
                break
        else:
            break  # No valid pairs left

        # Tie-breaking
        tied_pairs = [(neg_freq, pair)]
        while heap and -heap[0][0] == -neg_freq:
            neg_f2, p2 = heapq.heappop(heap)
            if p2 in pair2freq and -neg_f2 == pair2freq[p2]:
                tied_pairs.append((neg_f2, p2))

        # Choose lex largest among ties
        _, best_pair = max(tied_pairs, key=lambda x: x[1])
        for nf, p in tied_pairs:
            if p != best_pair:
                heapq.heappush(heap, (nf, p))

        # Double check by iterate over the pair2freq to find max freq and best pair
        # max_freq = max(pair2freq.values())
        # best_pair = max([p for p, f in pair2freq.items() if f == max_freq ])
        # assert max_freq == -neg_freq, "Max freq is not the same as the freq of the best pair"
        # assert best_pair == best_pair_bf, "Best pair is not the same as the best pair by brute force"
        # max_freq = max(pair2freq.values())
        # top_pairs = [pair for pair, freq in pair2freq.items() if freq == max_freq]
        # best_pair = max(top_pairs)

        merged_token = best_pair[0] + best_pair[1] # Concatenate two bytes object
        merges.append(best_pair)
        vocab[curr_id] = merged_token
        curr_id += 1

        # Update pretoken_freq with merged token
        new_freq = defaultdict(int)
        for token_seq, freq in pretoken_freq.items():
            # Process token_seq like classic BPE by maintaining a new_seq
            new_seq = []
            i = 0
            while i < len(token_seq):
                if (
                    i < len(token_seq) - 1
                    and token_seq[i] == best_pair[0]
                    and token_seq[i + 1] == best_pair[1]
                ):
                    # one new best pair disappear once merged
                    # all best pairs will be removed from pair2freq
                    # Hence, merged pair are unique
                    pair2freq[best_pair] -= freq
                    if pair2freq[best_pair] == 0:
                        del pair2freq[best_pair]
                    
                    if pair2freq[best_pair] > 0:
                        heapq.heappush(heap, (-pair2freq[best_pair], best_pair))

                    new_tok = token_seq[i] + token_seq[i + 1]
                    new_seq.append(new_tok)

                    # remove old pairs
                    if i > 0:
                        old_left = (token_seq[i - 1], token_seq[i])
                        pair2freq[old_left] -= freq

                        if pair2freq[old_left] == 0:
                            del pair2freq[old_left]

                        if pair2freq[old_left] > 0:
                            heapq.heappush(heap, (-pair2freq[old_left], old_left))

                        new_left = (token_seq[i - 1], new_tok)
                        pair2freq[new_left] += freq

                        heapq.heappush(heap, (-pair2freq[new_left], new_left))

                    if i + 2 < len(token_seq):
                        old_right = (token_seq[i + 1], token_seq[i + 2])
                        pair2freq[old_right] -= freq
                        if pair2freq[old_right] == 0:
                            del pair2freq[old_right]

                        if pair2freq[old_right] > 0:
                            heapq.heappush(heap, (-pair2freq[old_right], old_right))

                        new_right = (new_tok, token_seq[i + 2])
                        pair2freq[new_right] += freq

                        heapq.heappush(heap, (-pair2freq[new_right], new_right))

                    # skip the merged token
                    i += 2
                else:
                    new_seq.append(token_seq[i])
                    i += 1

            new_freq[tuple(new_seq)] += freq

        pretoken_freq = new_freq
    
    # add the unknown token
    # vocab[curr_id] = b"<|unk|>"

    return vocab, merges

def train_bpe_simple(pretoken_freq: dict[Tuple[bytes], int], vocab_size: int, special_tokens: list[str]):
    # Initialize vocabulary with special tokens and single-byte UTF-8
    vocab = {}
    curr_id = 0
    for token in special_tokens:
        vocab[curr_id] = token.encode("utf-8")
        curr_id += 1
    for i in range(256):
        vocab[curr_id] = bytes([i])
        curr_id += 1

    merges = []

    while len(vocab) < (vocab_size):
        # Count pair frequencies
        pair2freq = defaultdict(int)
        for token_seq, freq in pretoken_freq.items():
            for i in range(len(token_seq) - 1):
                pair = (token_seq[i], token_seq[i + 1])
                pair2freq[pair] += freq

        if not pair2freq:
            break  # No more pairs to merge

        # Select the most frequent pair (with lexicographical tiebreak)
        max_freq = max(pair2freq.values())
        top_pairs = [pair for pair, freq in pair2freq.items() if freq == max_freq]
        selected_pair = max(top_pairs)

        merges.append(selected_pair)
        merged_token = selected_pair[0] + selected_pair[1]
        vocab[curr_id] = merged_token
        curr_id += 1

        # Apply the merge to update pretoken_freq
        new_freq = defaultdict(int)
        for token_seq, freq in pretoken_freq.items():
            new_seq = []
            i = 0
            while i < len(token_seq):
                if (
                    i < len(token_seq) - 1
                    and token_seq[i] == selected_pair[0]
                    and token_seq[i + 1] == selected_pair[1]
                ):
                    new_seq.append(merged_token)
                    i += 2
                else:
                    new_seq.append(token_seq[i])
                    i += 1
            new_freq[tuple(new_seq)] += freq

        pretoken_freq = new_freq

    # add the unknown token
    # vocab[curr_id] = b"<|unk|>"

    return vocab, merges

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], **kwargs) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer on the given input path.
    """
    desired_num_chunks = 100
    # freq_table = pretokenize_multi_process(input_path, desired_num_chunks, special_tokens)
    freq_table = pretokenize_by_chunk_boundary_multi_process(input_path, desired_num_chunks, special_tokens)
    vocab, merges = train_bpe_heap_optimized(freq_table, vocab_size, special_tokens)
    return vocab, merges


class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]= None):
        ''' 
        This constructor should append special tokens to the vocab if they aren't in the vocab
        Note that key in vocab is int from 0 to vocab_size - 1
        '''
        self.vocab_reverse = {value: key for key, value in vocab.items()}
        if special_tokens is not None:
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in self.vocab_reverse.keys():
                    # extend the vocab with the special token
                    vocab[len(vocab)] = token.encode("utf-8")
                    self.vocab_reverse[token_bytes] = len(vocab)

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens 

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str]= None):
        with open(vocab_filepath, "r") as f:
            vocab = json.load(f)
        with open(merges_filepath, "r") as f:
            merges = json.load(f)
        # conver to bytes
        vocab = {int(key): val.encode("utf-8") for key, val in vocab.items()}
        merges = [(val[0].encode("utf-8"), val[1].encode("utf-8")) for val in merges]

        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        pretokens: list[list[bytes]] = self._pretokenize(text, keep_special_tokens=True)
        token_ids = []

        # merge pretokens according to BPE merges: list[tuple[bytes, bytes]]
        for merge in self.merges:
            new_pretokens = []
            for pretoken in pretokens:
                new_pretoken = []
                i = 0
                while i < len(pretoken):
                    if (i < len(pretoken) - 1
                        and pretoken[i] == merge[0]
                        and  pretoken[i + 1] == merge[1]):
                        new_pretoken.append(merge[0] + merge[1])

                        # skip the merged token
                        i += 2
                    else:
                        new_pretoken.append(pretoken[i])
                        i += 1
                new_pretokens.append(new_pretoken)
            pretokens = new_pretokens
            # print(f'pretokens: {pretokens}')
        
        # Scan vocab to find the token_id of merged pretoken
        for pretoken in pretokens:
            for token in pretoken:
                if token in self.vocab_reverse.keys():
                    token_ids.append(self.vocab_reverse[token])
                # else:
                    # token is not in vocab, it is a special token
                    # BPE traninig does not include special tokens, so we need to add them to the vocab
                    # This will increase the vocab size, but it is ok because we will not use the special tokens in the vocab
                    # Just ignore unknown tokens
                    # token_ids.append(len(self.vocab) - 1)

        return token_ids


    def big_text_iterable(self, input_path: str, desired_num_chunks: int = 100000) -> Iterable[str]:
        """
        Encode the text in memory efficient way.
        We have different ways to read the big text file by chunks:
            1. reading fixed size chunks in bytes
            2. chunking by special tokens like <|endoftext|>
            3. chunking by linek

        """
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
    
    def big_text_iterable_by_chunk_size(self, input_path: str, chunk_size : int = 1000) -> Iterable[str]:
        with open(input_path, encoding="utf-8") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    def encode_big_text_by_chunk_size(self, input_path: str, chunk_size: int = 1000) -> list[int]:
        '''
        input_path: str, path to the big text file
        chunk_size: int, max number of characters in the chunk to read
        return: list[int], token ids of the big text file
        '''
        big_text_iterable = self.big_text_iterable_by_chunk_size(input_path, chunk_size)

        token_ids = []

        current_chunkidx = 0
        for text in big_text_iterable:
            print(f'Start encoding chunk {current_chunkidx}......')
            token_ids.extend(self.encode(text))
            current_chunkidx += 1

            # if current_chunkidx == 2:
            #     break  # for testing
        
        return token_ids



    def encode_big_text(self, input_path: str, desired_num_chunks: int = 100000) -> list[int]:
        """
        Encode a big text file
        """
        big_text_iterable = self.big_text_iterable(input_path, desired_num_chunks)
        big_text_iterable = self.big_text_iterable(input_path, desired_num_chunks)

        # use np.memmap to store token_ids the max len is the number of bytes in the text file
        max_len = os.path.getsize(input_path)
        token_ids = np.memmap(input_path.replace(".txt", f"_{desired_num_chunks}_encoded.npy"), dtype=np.int32, mode='w+', shape=(max_len,))
        current_idx = 0
        used_len = 0
        for text in big_text_iterable:
            print(f'Start encoding chunk ......')
            token_ids_chunk = self.encode(text)
            token_ids[current_idx:current_idx + len(token_ids_chunk)] = token_ids_chunk
            current_idx += len(token_ids_chunk)
            used_len += len(token_ids_chunk)
            # token_ids.extend(token_ids_chunk)
            print(f'One chunk tokenization done')
        
        # resize the memmap to the used length
        token_ids.resize(used_len)

        return token_ids



    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        iterable is a python handle to a large text file
        """
        token_ids = []
        for text in iterable:
            token_ids.extend(self.encode(text))
        return token_ids

    def decode(self, ids: list[int]) -> str:
        """
        Note we encode in bytes level, so we need to decode in bytes level
        Decode one byte at a time is wrong, because the bytes object is not a string
        """
        # Recover the text bytes from token ids
        text_bytes = bytes()
        vocab_size = len(self.vocab)
        replacement_char = "\uFFFD"
        for token_id in ids: 
            # We assume token_id is from 0 to vocab_size - 1
            if token_id < vocab_size:
                text_bytes += self.vocab[token_id]
            else:
                text_bytes += bytes(replacement_char, encoding='utf-8')
        
        # Decode the text bytes to string
        return text_bytes.decode(encoding='utf-8', errors='replace')
   
    def _pretokenize(self, text: str, keep_special_tokens: bool = True) -> list[list[bytes]]:
        segments = self._split_by_special_tokens(text, self.special_tokens)
        pretokens = []
        for part in segments:
            if self.special_tokens is not None and part in self.special_tokens and not keep_special_tokens:
                continue
            elif self.special_tokens is not None and part in self.special_tokens and keep_special_tokens:
                # note encode() returns a bytes object whose value is the explaination of bytes in memory
                # wrap part as one element tuple 
                pretokens.append([part.encode("utf-8")])
                continue
            else:   
                # Process non-special tokens
                for match in re.finditer(BPETokenizer.PAT, part):
                    string = match.group()
                    # Here the tuple of bytes integer value acts as key---pre-token is a tuple of bytes
                    # encode() returns a bytes object whose value is the explaination of bytes in memory
                    # integer is the unicode value of the string encoded by utf-8
                    # bytes[integer] is a byte instance of the string with value as string's encoded value
                    pretokens.append([bytes([integer]) for integer in list(string.encode("utf-8"))])
        
        return pretokens

    def _split_by_special_tokens(self, text: str, special_tokens: list[str]) -> list[str]:
        """
        Split on the special tokens
        example: 
            text = "Hello world! <|endoftext|> Great!" 
            special_tokens = "<|endoftext|>"
            result = ['Hello world! ', '<|endoftext|>', ' Great!']
        """
        if not special_tokens:
            parts = [text]
        else:
            # escape special meaning of speical tokens for regex 
            pattern = "|".join(re.escape(tok) for tok in special_tokens)
            # wrap pattern with () to make it a group in the final regex results
            parts = re.split('(' + pattern + ')', text)

        # print(f'parts: {parts}')
        return parts

        
if __name__ == "__main__":
    BPE = BPETokenizer.from_files("vocab.json", "merges.json", ["<|endoftext|>"])
    # text = "Hello, world! Hello,你好 world! <|endoftext|> It's a beautiful day."
    text = "hello world <|endoftext|>, I love you so much!!"
    # text = ""
    # text = "🙃"
    pretokens = BPE._pretokenize(text, keep_special_tokens=True)
    print(pretokens)
    token_ids = BPE.encode(text)
    print(f'token_ids: {token_ids}')
    decoded_text = BPE.decode(token_ids)
    print(f'decoded_text: {decoded_text}')
    print(f'is same: {text == decoded_text}')


    # file_path = "/home/harveyai/Code/llm_scratch/assignment1-basics/cs336_basics/data/corpus.en"
    # file_path = "/home/harveyai/Code/llm_scratch/assignment1-basics/cs336_basics/data/TinyStoriesV2-GPT4-train.txt"
    # desired_num_chunks = 10000
    # doc_sep_token_bytes = b"<|endoftext|>"
    # mini_chunk_size = 4096
    # # include all whitespace tokens
    # special_tokens_list = ["<|endoftext|>"]


    # # text = "Hello, world! Hello, world! It's a beautiful day."
    # freq_table = pretokenize_by_chunk_boundary_multi_process(file_path, desired_num_chunks, special_tokens_list)
    # # print(freq_table)
    # start_time = time.time()
    # vocab_simple, merges_simple = train_bpe_simple(freq_table, 500, special_tokens_list)
    # end_time = time.time()
    # # print(f'vocab: {vocab_simple}')
    # # print(f'merges: {merges_simple}')
    # print(f'time: {end_time - start_time}')

    # # compare with heap optimized
    # start_time = time.time()
    # vocab, merges = train_bpe_heap_optimized(freq_table, 500, special_tokens_list)
    # end_time = time.time()
    # # print(f'vocab: {vocab}')
    # # print(f'merges: {merges}')
    # print(f'time: {end_time - start_time}')

    # # difference between heap optimized and simple
    # is_same = merges == merges_simple
    # print(f'is same: {is_same}')

    # vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]
    # vocab, merges = train_bpe(file_path, 500, special_tokens_list)
    # # convert bytes to string
    # vocab = {key: val.decode('utf-8', errors='ignore') for key, val in vocab.items()}
    # merges = [(val[0].decode('utf-8', errors='ignore'), val[1].decode('utf-8', errors='ignore')) for val in merges]
    # # save vocab and merges to json after converting bytes to string
    # with open("vocab.json", "w") as f:
    #     json.dump(vocab, f)
    # with open("merges.json", "w") as f:
    #     json.dump(merges, f)

    # with open(file_path, "rb") as f:
    #     chunk_boundaries = find_chunk_boundaries(f, desired_num_chunks, doc_sep_token_bytes, mini_chunk_size)
    #     chunk_boundaries_single_process = find_chunk_boundaries_single_process(f, desired_num_chunks, doc_sep_token_bytes)

    #     print(f'chunk_boundaries: {chunk_boundaries}')
    #     print(f'chunk_boundaries_single_process: {chunk_boundaries_single_process}')
    #     print(f'is same: {chunk_boundaries == chunk_boundaries_single_process}')
