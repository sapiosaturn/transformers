# NOTE: training with random placeholder dataset to begin with

import torch
import torch.optim as optim
import torch.nn.functional as F

from models import DecoderOnlyTransformer

TRAINING_CONFIG = {
    "device": "cpu",
    "batch_size": 4,
    "learning_rate": 0.01,
    "num_epochs": 10
}

MODEL_CONFIG = {
    "vocab_size": 26,
    "num_layers": 3,
    "embedding_dim": 8,
    "num_heads": 2,
    "context_length": 4,
    "feedforward_dim": 16,
    "attention_dropout_p": 0,
    "residual_dropout_p": 0
}

model = DecoderOnlyTransformer(
    vocab_size = MODEL_CONFIG["vocab_size"],
    num_layers = MODEL_CONFIG["num_layers"],
    embedding_dim = MODEL_CONFIG["embedding_dim"],
    num_heads = MODEL_CONFIG["num_heads"],
    context_length = MODEL_CONFIG["context_length"],
    feedforward_dim = MODEL_CONFIG["feedforward_dim"],
    attention_dropout_p = MODEL_CONFIG["attention_dropout_p"],
    residual_dropout_p = MODEL_CONFIG["residual_dropout_p"]
)

