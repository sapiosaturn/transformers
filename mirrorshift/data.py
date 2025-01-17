"""
This file will contain a bunch of abstractions for simplifying interacting with datasets.
"""

import torch
import tiktoken
from torch.utils.data import Dataset

class CharacterTxtDataset(Dataset):
    """
    Simple dataset to use if the whole dataset is in a text file, character-level tokenizer.
    Data is all loaded into memory.
    Reference: https://gist.github.com/karpathy/d4dee566867f8291f086
    """

    def __init__(self, file_path: str, sequence_length: int) -> None:
        self.file_path = file_path
        self.sequence_length = sequence_length
        with open(file_path, 'r') as f:
            self.data = f.read()
        self.vocab = sorted(list(set(self.data))) # sorted here so indexes are deterministic
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(self.vocab)}

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def tokenize(self, sequence: str) -> torch.Tensor:
        return torch.tensor([self.char_to_idx[c] for c in sequence], dtype=torch.long)

    def detokenize(self, token_ids: list[int]) -> str:
        return ''.join([self.idx_to_char[int(id)] for id in token_ids])

    def __len__(self) -> int:
        return len(self.data) - self.sequence_length - 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # returned sequences should be sequences of token IDs
        x = self.tokenize(self.data[index:index+self.sequence_length])
        y = self.tokenize(self.data[index+1:index+self.sequence_length+1]) # shifting x sequence by one
        return x, y

class TiktokenTxtDataset(Dataset):
    def __init__(self, file_path: str, sequence_length: int, tokenizer_name: str = "p50k_base") -> None:
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)

        # load file and tokenize all at once, whole dataset in memory
        with open(file_path, 'r') as f:
            text = f.read()
        self.tokens = self.tokenizer.encode(text)

    def __len__(self) -> int:
        return len(self.tokens) - self.sequence_length - 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[idx : idx + self.sequence_length]
        y = self.tokens[idx + 1 : idx + self.sequence_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def detokenize(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def get_vocab_size(self) -> int:
        return self.tokenizer.n_vocab
