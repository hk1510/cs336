from bpe_tokenizer import BPETokenizer
import time
import pickle

if __name__ == "__main__":
    tokenizer = BPETokenizer(vocab_size=10000, special_tokens=["<|endoftext|>"])

    # vocab, merges = tokenizer.train("data/TinyStoriesV2-GPT4-train.txt")
    start_time = time.time()
    vocab, merges = tokenizer.train_bpe(
        "data/TinyStoriesV2-GPT4-train.txt", num_processes=4
    )
    end_time = time.time()

    print("Time Taken:", end_time - start_time)
    with open("vocab_and_merges.pkl", "wb") as f:
        pickle.dump((vocab, merges), f)
