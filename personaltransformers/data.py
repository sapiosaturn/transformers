"""
This file will contain a bunch of abstractions for simplifying interacting with datasets.
"""

import torch
from torch.utils.data import DataLoader, Dataset

class TxtFileDataset(Dataset):
    """
    Simple dataset to use if the whole dataset is in a text file, character-level tokenizer.
    Data is all loaded into memory.
    Reference: https://gist.github.com/karpathy/d4dee566867f8291f086
    """

    def __init__(self, file_path, sequence_length):
        self.file_path = file_path
        self.sequence_length = sequence_length
        with open(file_path, 'r') as f:
            self.data = f.read()
        self.vocab = sorted(list(set(self.data))) # sorted here so indexes are deterministic
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(self.vocab)}

    def get_vocab_size(self):
        return len(self.vocab)

    def tokenize(self, sequence):
        return torch.tensor([self.char_to_idx[c] for c in sequence], dtype=torch.long)

    def detokenize(self, token_ids):
        return ''.join([self.idx_to_char[int(id)] for id in token_ids])

    def __len__(self):
        return len(self.data) - self.sequence_length - 1

    def __getitem__(self, index):
        # returned sequences should be sequences of token IDs
        x = self.tokenize(self.data[index:index+self.sequence_length])
        y = self.tokenize(self.data[index+1:index+self.sequence_length+1]) # shifting x sequence by one
        return x, y
