"""
This file contains the core training logic.
"""

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import argparse
from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from models import CausalTransformer
from data import TiktokenTxtDataset
from inference import sample
from utils import read_model_config, read_training_config, ModelConfig, TrainingConfig

BatchType = Tuple[torch.Tensor, torch.Tensor]
Logits = torch.Tensor

def sample_and_log(
    model: CausalTransformer,
    global_step: int,
    writer: SummaryWriter,
    device: str,
    dataset: Dataset,
    model_config: ModelConfig,
    sampling_length: int
) -> None:
    model.eval()
    random_token = torch.randint(
        low=0, high=model_config.vocab_size, size=(1, 1)
    )
    tokens, generated_text = sample(
        model=model,
        context=random_token,
        num_tokens=sampling_length,
        context_length=model_config.context_length,
        device=device,
        dataset=dataset,
    )
    print("\n╭─ Generated Text Sample ──────────────")
    print("│")
    for line in generated_text.split('\n'):
        print(f"│ {line}")
    print("│")
    print("╰──────────────────────────────────────")
    writer.add_text("Sampled Text", generated_text, global_step)
    model.train()

def train(
    model: CausalTransformer,
    train_loader: DataLoader,
    opt: optim.AdamW,
    scheduler: lr_scheduler.LinearLR,
    device: str,
    training_config: TrainingConfig,
    model_config: ModelConfig,
    writer: SummaryWriter,
) -> None:
    global_step: int = 0

    for e in range(training_config.num_epochs):
        print("\n╭─ Starting Epoch ─────────────────────")
        print(f"│ Epoch: {e}")
        print("╰──────────────────────────────────────")
        running_loss: float = 0.0
        for i, batch in enumerate(train_loader):
            x: torch.Tensor
            y: torch.Tensor
            batch: BatchType = batch
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits: Logits = model(x)

            loss_value = F.cross_entropy(
                logits.view(logits.size(0) * logits.size(1), logits.size(2)),
                y.view(y.size(0) * y.size(1)),
            )
            loss_value.backward()
            opt.step()
            scheduler.step()
            global_step += 1

            loss_scalar = loss_value.item()
            writer.add_scalar("Loss/train", loss_scalar, global_step)
            running_loss += loss_scalar

            if i % training_config.reporting_factor == training_config.reporting_factor - 1:
                last_loss = running_loss / training_config.reporting_factor
                print("\n╭─ Training Progress ──────────────────")
                print(f"│ Batch Size: {training_config.batch_size}")
                print(f"│ Step:       {i+1}")
                print(f"│ Loss:       {last_loss:.5f}")
                print("╰──────────────────────────────────────")
                running_loss = 0.0

            if i % training_config.sampling_factor == training_config.sampling_factor - 1:
                sample_and_log(
                    model=model,
                    global_step=global_step,
                    writer=writer,
                    device=device,
                    dataset=train_loader.dataset,
                    model_config=model_config,
                    sampling_length=training_config.sampling_length_multiplier * model_config.context_length
                )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config', type=str, default='config/model_configs/ss_small.json',
                      help='Path to model configuration file')
    parser.add_argument('--training-config', type=str, default='config/training_configs/ss_small.json',
                      help='Path to training configuration file')
    parser.add_argument('--dataset', type=str, default='./datasets/tinyshakespeare.txt',
                      help='Path to training dataset file')
    args = parser.parse_args()

    model_config: ModelConfig = read_model_config(args.model_config)
    training_config: TrainingConfig = read_training_config(args.training_config)

    writer: SummaryWriter = SummaryWriter()

    dataset: TiktokenTxtDataset = TiktokenTxtDataset(
        args.dataset, model_config.context_length
    )

    assert dataset.get_vocab_size() == model_config.vocab_size, \
    f"dataset vocab size is {dataset.get_vocab_size()}, model_config vocab size is {model_config.vocab_size}"

    model: CausalTransformer = CausalTransformer(model_config=model_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model has {trainable_params} trainable parameters")

    device: str
    if training_config.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            torch.set_float32_matmul_precision("high")
        elif torch.backends.mps.is_available() and not training_config.compile:
            device = "mps"
        else:
            device = "cpu"
    else:
        device = training_config.device
    model = model.to(device)
    print("running on device: ", device)

    opt: optim.AdamW = optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    scheduler: lr_scheduler.LinearLR = lr_scheduler.LinearLR(
        opt, 0.1, 1, training_config.lr_warmup_steps
    )

    model = torch.compile(model) if training_config.compile else model

    train_loader: DataLoader = DataLoader(
        dataset, batch_size=training_config.batch_size, shuffle=True
    )

    train(
        model=model,
        train_loader=train_loader,
        opt=opt,
        scheduler=scheduler,
        device=device,
        training_config=training_config,
        model_config=model_config,
        writer=writer,
    )

    writer.flush()
