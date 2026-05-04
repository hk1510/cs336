import regex as re
from typing import BinaryIO
from .pretokenization_example import find_chunk_boundaries
from collections import defaultdict

class BPETokenizer:

    def __init__(self, vocab_size: int, special_tokens: list[str]):

        self.vocab = special_tokens + [bytes([i]) for i in range(256)]

        min_vocab_size = len(self.vocab)

        if vocab_size <= min_vocab_size:
            raise ValueError(f"Vocab size is less than the minimum of 256 + {len(special_tokens)} = {min_vocab_size}")

        self.vocab_size = vocab_size

        self.pretokenize_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" 

    def tokenize(self, file_path: str):

        with open(file_path, "rb") as f:
            num_processes = 4
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            pretok_dict = defaultdict(int)

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                chunk_pretok_dict = defaultdict(int)
                for pretok in re.finditer(self.pretokenize_pattern, chunk):
                    chunk_pretok_dict[tuple(pretok.group().encode())] += 1
                for key in chunk_pretok_dict:
                    pretok_dict[key] += chunk_pretok_dict[key]

            print(pretok_dict)

        
if __name__ == "__main__":
    tokenizer = BPETokenizer(vocab_size=400, special_tokens=['<|endoftext|>'])
    tokenizer.tokenize('test.txt')
