# NOTE: training with placeholder dataset to begin with

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import DecoderOnlyTransformer
from data import TxtFileDataset

TRAINING_CONFIG = {
    "device": "auto",
    "batch_size": 16,
    "learning_rate": 1e-7,
    "num_epochs": 100
}

MODEL_CONFIG = {
    "vocab_size": 65,
    "num_layers": 6,
    "embedding_dim": 64,
    "num_heads": 4,
    "context_length": 256,
    "feedforward_dim": 128,
    "attention_dropout_p": 0.0,
    "residual_dropout_p": 0.0
}

dataset = TxtFileDataset(r'./datasets/tinyshakespeare.txt', MODEL_CONFIG["context_length"])
assert dataset.get_vocab_size() == MODEL_CONFIG["vocab_size"]

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

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"model has {trainable_params} trainable parameters")

# determine the device we'll train on
if TRAINING_CONFIG["device"] == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = TRAINING_CONFIG["device"]
model = model.to(device)
print("running on device: ", device)

opt = optim.AdamW(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])

@torch.compile(fullgraph=False)
def opt_step():
    opt.step

model = torch.compile(model)
train_loader = DataLoader(dataset, batch_size=TRAINING_CONFIG["batch_size"], shuffle=True)

for e in range(TRAINING_CONFIG["num_epochs"]):
    print(f"epoch {e} —")
    running_loss = 0.
    for i, batch in enumerate(train_loader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        logits = model(x)
        # cross entropy expects (pseudo_batch_size, number_of_classes), so we reshape the logits into that
        # y is shaped like (pseudo_batch_size) since the target is the "correct" class
        # important to note here that pseudo_batch_size = batch_size * sequence_length
        # logits are batch_size, sequence_length, vocab_size
        pseudo_batch_size = logits.size(0)*logits.size(1)
        loss_value = F.cross_entropy(logits.view(logits.size(0)*logits.size(1), logits.size(2)), y.view(y.size(0)*y.size(1)))
        loss_value.backward()
        opt.step()

        running_loss += loss_value.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print(f"  batch {i+1} loss: {last_loss}")
            running_loss = 0.

