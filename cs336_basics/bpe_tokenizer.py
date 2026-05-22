import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries
from collections import defaultdict
from multiprocessing import Process, Queue
from typing import Protocol, TypeAlias, cast
import mmap
import os
import pickle
import time

PretokenCounts: TypeAlias = dict[tuple[bytes, ...], int]


class PretokenCountsQueue(Protocol):
    def put(
        self, obj: PretokenCounts, block: bool = True, timeout: float | None = None
    ) -> None: ...
    def get(self) -> PretokenCounts: ...


class BPETokenizer:
    def __init__(self, vocab_size: int, special_tokens: list[str]):

        self.special_tokens: list[str] = special_tokens

        self.vocab: dict[int, bytes] = {}

        for i in range(256):
            self.vocab[i] = bytes([i])

        min_vocab_size = len(self.vocab)

        if vocab_size <= min_vocab_size:
            raise ValueError(
                f"Vocab size is less than the minimum of 256 + {len(special_tokens)} = {min_vocab_size}"
            )

        self.vocab_size: int = vocab_size

        self.pretokenize_pattern: str = (
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        self.counts_cache: dict[
            tuple[bytes, bytes], list[dict[tuple[bytes, ...], int]]
        ] = {}

    def create_byte_pair_dict(self, pretok_dict: dict[tuple[bytes, ...], int]) -> tuple[
        dict[tuple[bytes, bytes], int],
        dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    ]:
        byte_pair_dict: dict[tuple[bytes, bytes], int] = defaultdict(int)
        byte_pair_pretok_dict: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = (
            defaultdict(set)
        )
        for key in pretok_dict:
            for i in range(len(key) - 1):
                pair = (key[i], key[i + 1])
                byte_pair_dict[pair] += pretok_dict[key]
                byte_pair_pretok_dict[pair].add(key)
        return byte_pair_dict, byte_pair_pretok_dict

    def merge(
        self,
        pretok_dict: dict[tuple[bytes, ...], int],
        byte_pair_dict: dict[tuple[bytes, bytes], int],
        byte_pair_pretok_dict: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    ) -> tuple[bytes, bytes] | None:
        if len(byte_pair_dict) == 0:
            return None

        max_pair = max(
            byte_pair_dict.items(), key=lambda item: (item[1], *item[0])
        )

        merge_pair = max_pair[0]
        new_token = merge_pair[0] + merge_pair[1]

        self.vocab[len(self.vocab)] = new_token
        del byte_pair_dict[merge_pair]
        for pretok in byte_pair_pretok_dict[merge_pair]:
            new_key: list[bytes] | tuple[bytes, ...] = []
            merged = False
            modified_index_pairs: set[tuple[int, int]] = set()
            for i in range(len(pretok)):
                if merged == True:
                    merged = False
                    continue
                if i == len(pretok) - 1:
                    new_key.append(pretok[i])
                    continue
                if merge_pair[0] == pretok[i] and merge_pair[1] == pretok[i + 1]:
                    merged = True
                    new_key.append(new_token)

                    if i > 0:
                        left_pair = (pretok[i - 1], pretok[i])
                        if (i - 1, i) not in modified_index_pairs:
                            byte_pair_dict[left_pair] -= pretok_dict[pretok]
                            if byte_pair_dict[left_pair] <= 0:
                                del byte_pair_dict[left_pair]
                            modified_index_pairs.add((i-1, i))
                    if i < len(pretok) - 2:
                        right_pair = (pretok[i + 1], pretok[i + 2])
                        if (i + 1, i + 2) not in modified_index_pairs:
                            byte_pair_dict[right_pair] -= pretok_dict[pretok]
                            if byte_pair_dict[right_pair] <= 0:
                                del byte_pair_dict[right_pair]
                            modified_index_pairs.add((i + 1, i + 2))
                else:
                    new_key.append(pretok[i])

            new_key = tuple(new_key)
            pretok_dict[new_key] += pretok_dict[pretok]
            del pretok_dict[pretok]

            for i in range(len(new_key)):
                if new_key[i] == new_token:
                    if i > 0:
                        byte_pair_dict[(new_key[i - 1], new_token)] += pretok_dict[
                            new_key
                        ]
                    if i < len(new_key) - 1:
                        byte_pair_dict[(new_token, new_key[i + 1])] += pretok_dict[
                            new_key
                        ]
                if i < len(new_key) - 1:
                    byte_pair_pretok_dict[(new_key[i], new_key[i + 1])].add(new_key)
                    if pretok in byte_pair_pretok_dict[(new_key[i], new_key[i + 1])]:
                        byte_pair_pretok_dict[(new_key[i], new_key[i + 1])].remove(
                            pretok
                        )

        return merge_pair

    def pretokenize_chunk(
        self, queue: PretokenCountsQueue, start: int, end: int, file_name: str
    ):
        with open(file_name, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            _ = mm.seek(start)
            chunk: str = mm.read(end - start).decode("utf-8", errors="ignore")
            sub_chunks = re.split(
                "|".join([re.escape(x) for x in self.special_tokens]), chunk
            )
            chunk_pretok_dict: PretokenCounts = defaultdict(int)
            for sub_chunk in sub_chunks:
                #
                for pretok in re.finditer(self.pretokenize_pattern, sub_chunk):
                    byte_tuple: tuple[bytes, ...] = tuple(
                        bytes([i]) for i in pretok.group().encode()
                    )
                    if len(byte_tuple) > 0:
                        chunk_pretok_dict[byte_tuple] += 1
            queue.put(chunk_pretok_dict)

    def train_bpe(
        self, file_path: str | os.PathLike, num_processes = 8,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

        with open(file_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        pretok_dict: dict[tuple[bytes, ...], int] = defaultdict(int)
        boundary_tuples: list[tuple[int, int]] = list(
            zip(boundaries[:-1], boundaries[1:])
        )
        num_chunks: int = len(boundary_tuples)
        q: PretokenCountsQueue = cast(PretokenCountsQueue, Queue())
        processes: list[Process] = []

        for i in range(num_chunks):
            p: Process = Process(
                target=self.pretokenize_chunk,
                args=(q, boundary_tuples[i][0], boundary_tuples[i][1], file_path),
            )
            p.start()
            processes.append(p)

        for _ in range(num_chunks):
            chunk_pretok_dict: PretokenCounts = q.get()
            for key in chunk_pretok_dict:
                pretok_dict[key] += chunk_pretok_dict[key]

        for i in range(num_chunks):
            processes[i].join()

        print("Pretokenization complete")

        byte_pair_dict, byte_pair_pretok_dict = self.create_byte_pair_dict(pretok_dict)

        merges: list[tuple[bytes, bytes]] = []
        while len(self.vocab) < self.vocab_size - len(self.special_tokens):
            merge = self.merge(pretok_dict, byte_pair_dict, byte_pair_pretok_dict)
            if merge is None:
                break
            merges.append(merge)

        vocab_length = len(self.vocab)
        for i, token in enumerate(self.special_tokens):
            self.vocab[vocab_length + i] = token.encode("utf-8")

        return self.vocab, merges


if __name__ == "__main__":
    tokenizer = BPETokenizer(vocab_size=32000, special_tokens=["<|endoftext|>"])
    # vocab, merges = tokenizer.train("data/TinyStoriesV2-GPT4-train.txt")
    start_time = time.time()
    vocab, merges = tokenizer.train_bpe("data/owt_train.txt")
    end_time = time.time()
    print("Time Taken:", end_time - start_time)
    with open("owt_vocab_and_merges.pkl", "wb") as f:
        pickle.dump((vocab, merges), f)
