import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries
from collections import defaultdict
from multiprocessing import Process, Queue
from typing import Protocol, TypeAlias, cast
import mmap

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
            self.vocab[i + len(special_tokens)] = bytes([i])

        min_vocab_size = len(self.vocab)

        if vocab_size <= min_vocab_size:
            raise ValueError(
                f"Vocab size is less than the minimum of 256 + {len(special_tokens)} = {min_vocab_size}"
            )

        self.vocab_size: int = vocab_size

        self.pretokenize_pattern: str = (
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        self.counts_cache: dict[tuple[bytes, bytes], list[dict[tuple[bytes, ...], int]]]= {}

    # {lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}
    #

    def create_byte_pair_dict(self, pretok_dict: dict[tuple[bytes, ...], int]) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
        byte_pair_dict: dict[tuple[bytes, bytes], int] = defaultdict(int)
        byte_pair_pretok_dict: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
        for key in pretok_dict:
            for i in range(len(key) - 1):
                pair = (key[i], key[i + 1])
                byte_pair_dict[pair] += pretok_dict[key]
                byte_pair_pretok_dict[pair].add(key)
        return byte_pair_dict, byte_pair_pretok_dict

    def merge(
            self, pretok_dict: dict[tuple[bytes, ...], int], byte_pair_dict: dict[tuple[bytes, bytes], int], byte_pair_pretok_dict: dict[tuple[bytes, bytes], set[tuple[bytes,...]]]
    ) -> tuple[bytes, bytes] | None:
        sorted_pairs = sorted(
            byte_pair_dict.items(), key=lambda item: (item[1], *item[0]), reverse=True
        )
        if len(sorted_pairs) == 0:
            return None

        merge_pair = sorted_pairs[0][0]

        print("MERGING: ", merge_pair[0], merge_pair[1])
        self.vocab[len(self.vocab) + len(self.special_tokens)] = (
            merge_pair[0] + merge_pair[1]
        )
       
        # TODO: Add merged pair + pretok counts to byte pair dict
        # TODO: Subtract pretok counts from bytes that occur before and after merged pair
        for pretok in byte_pair_pretok_dict[merge_pair]:
            new_key: list[bytes] | tuple[bytes, ...] = []
            contains_merge = False
            merged = False
            for i in range(len(pretok)):
                if merged == True:
                    merged = False
                    continue
                if i == len(pretok) - 1:
                    new_key.append(pretok[i])
                    continue
                if merge_pair[0] == pretok[i] and merge_pair[1] == pretok[i + 1]:
                    merged = True
                    contains_merge = True
                    new_key.append(merge_pair[0] + merge_pair[1])
                    if i > 0:
                        byte_pair_dict[(pretok[i-1], pretok[i])] -= pretok_dict[pretok]
                    

                else:
                    new_key.append(pretok[i])
            new_key = tuple(new_key)
            if contains_merge:
                pretok_dict[new_key] = pretok_dict[pretok]
                del pretok_dict[pretok]
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
                for pretok in re.finditer(self.pretokenize_pattern, sub_chunk):
                    byte_tuple: tuple[bytes, ...] = tuple(
                        bytes([i]) for i in pretok.group().encode()
                    )
                    if len(byte_tuple) > 0:
                        chunk_pretok_dict[byte_tuple] += 1
            queue.put(chunk_pretok_dict)

    def train(
        self, file_path: str
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

        with open(file_path, "rb") as f:
            num_processes = 8
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
        
        print('Pretokenization complete')

        byte_pair_dict, byte_pair_pretok_dict = self.create_byte_pair_dict(pretok_dict)

        merges: list[tuple[bytes, bytes]] = []
        while len(self.vocab) < self.vocab_size:
            merge = self.merge(pretok_dict, byte_pair_dict, byte_pair_pretok_dict)
            if merge:
                merges.append(merge)
        return self.vocab, merges


if __name__ == "__main__":
    tokenizer = BPETokenizer(vocab_size=10000, special_tokens=["<|endoftext|>"])
    vocab, merges = tokenizer.train("data/TinyStoriesV2-GPT4-train.txt")
    for merge in merges:
        print([x for x in merge])

    print(vocab)
