import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries
from collections import defaultdict


class BPETokenizer:
    def __init__(self, vocab_size: int, special_tokens: list[str]):

        self.vocab: list[str | bytes] = special_tokens + [
            bytes([i]) for i in range(256)
        ]

        min_vocab_size = len(self.vocab)

        if vocab_size <= min_vocab_size:
            raise ValueError(
                f"Vocab size is less than the minimum of 256 + {len(special_tokens)} = {min_vocab_size}"
            )

        self.vocab_size: int = vocab_size

        self.pretokenize_pattern: str = (
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    def merge(
        self, pretok_dict: dict[tuple[bytes, ...], int]
    ) -> tuple[bytes, bytes] | None:
        byte_pair_dict: dict[tuple[bytes, bytes], int] = defaultdict(int)
        for key in pretok_dict:
            for i in range(len(key) - 1):
                pair = (key[i], key[i + 1])
                byte_pair_dict[pair] += pretok_dict[key]
        sorted_pairs = sorted(
            byte_pair_dict.items(), key=lambda item: (item[1], *item[0]), reverse=True
        )
        if len(sorted_pairs) == 0:
            return None

        merge_pair = sorted_pairs[0][0]

        print("MERGING: ", merge_pair[0], merge_pair[1])
        self.vocab.append((merge_pair[0].decode() + merge_pair[1].decode()).encode())

        for key in list(pretok_dict.keys()):
            new_key: list[bytes] | tuple[bytes, ...] = []
            contains_merge = False
            merged = False
            for i in range(len(key)):
                if merged == True:
                    merged = False
                    continue
                if i == len(key) - 1:
                    new_key.append(key[i])
                    continue
                if merge_pair[0] == key[i] and merge_pair[1] == key[i + 1]:
                    merged = True
                    contains_merge = True
                    new_key.append(
                        (merge_pair[0].decode() + merge_pair[1].decode()).encode()
                    )
                else:
                    new_key.append(key[i])
            new_key = tuple(new_key)
            if contains_merge:
                pretok_dict[new_key] = pretok_dict[key]
                del pretok_dict[key]
        return merge_pair

    def tokenize(self, file_path: str):

        with open(file_path, "rb") as f:
            num_processes = 4
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            pretok_dict: dict[tuple[bytes, ...], int] = defaultdict(int)

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                _ = f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                chunk_pretok_dict: dict[tuple[bytes, ...], int] = defaultdict(int)
                # re.finditer(self.pretokenize_pattern, chunk)
                for pretok in re.split(r"[ \n]+", chunk):
                    # tuple(pretok.group().encode())
                    if pretok != "":
                        chunk_pretok_dict[
                            tuple(bytes([i]) for i in pretok.encode())
                        ] += 1
                for key in chunk_pretok_dict:
                    pretok_dict[key] += chunk_pretok_dict[key]

            print(pretok_dict)

        merges = []
        while len(self.vocab) < self.vocab_size:
            merges.append(self.merge(pretok_dict))
        return self.vocab, merges


if __name__ == "__main__":
    tokenizer = BPETokenizer(vocab_size=256 + 1 + 6, special_tokens=["<|endoftext|>"])
    vocab, merges = tokenizer.tokenize("test.txt")
    for merge in merges:
        print([x.decode() for x in merge])

    print(vocab)
