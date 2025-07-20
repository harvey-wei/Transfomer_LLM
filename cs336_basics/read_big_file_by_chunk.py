import os
import numpy as np


def read_big_file_by_chunk(file_path: str, chunk_size: int = 1000):
    '''
    file_path: str, path to the file to read
    chunk_size: int, maximum count of characters in the chunk to read
    '''
    with open(file_path, encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                # empty string means end of file
                break
            yield chunk

def test_read_big_file_by_chunk():
    file_path = "data/TinyStoriesV2-GPT4-valid.txt"
    chunk_size = 1000 # 1000 characters
    for chunk in read_big_file_by_chunk(file_path, chunk_size):
        # check whether the size of the chunk is equal to chunk_size in bytes       
        print(chunk)

        if len(chunk) == chunk_size:
            print(f'curent chunk has {len(chunk)} characters matching chunk_size')
        elif len(chunk) < chunk_size:
            print(len(chunk), " current chunk has less than chunk_size chars.")
        else:
            print(len(chunk), " current chunk has more than chunk_size chars.")
            break

def load_big_file_as_mmap(file_path: str):
    '''
    file_path: str, path to the file to read
    ✅ What format does np.memmap load a file in?
    np.memmap maps a file on disk into a NumPy array. It does not load the entire file into memory, but treats the file as if it were a NumPy array.

    🔧 Format:
    It interprets the file as a raw binary array of elements of a given dtype.
    '''
    file_size = os.path.getsize(file_path) # get the size of the file in bytes
    return np.memmap(file_path, dtype=np.uint8, mode='r+', shape=(file_size,))

def test_load_big_file_by_mmap():
    file_path = "data/TinyStoriesV2-GPT4-valid.txt"
    mm = load_big_file_as_mmap(file_path) # list of byte uint8
    chunk = mm[:100]
    print(f'mm in raw format: {chunk}')
    # print(f'mm in bytes format: {chunk.tobytes()}')
    print(f'mm in string format: {chunk.tobytes().decode("utf-8")}')


if __name__ == "__main__":
    # test_read_big_file_by_chunk()
    test_load_big_file_by_mmap()