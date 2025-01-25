"""
This file contains the core training logic.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import argparse
from typing import Tuple
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    Subset
)
from torch.utils.tensorboard import SummaryWriter

from models import CausalTransformer
from data import TiktokenTxtDataset
from inference import sample
from utils import (
    read_model_config,
    read_training_config,
    ModelConfig,
    TrainingConfig,
    get_lr_schedule,
    get_supported_dtype
)
from distributed import (
    setup_distributed,
    fsdp_wrap,
    cleanup_distributed
)

BatchType = Tuple[torch.Tensor, torch.Tensor]
Logits = torch.Tensor

def sample_and_log(
    model: CausalTransformer,
    global_step: int,
    writer: SummaryWriter,
    device: str,
    subset: Subset,
    model_config: ModelConfig,
    sampling_length: int
) -> None:
    model.eval()
    random_token = torch.randint(
        low=0, high=model_config.vocab_size, size=(1, 1)
    )
    model_dtype = next(model.parameters()).dtype
    with torch.autocast(device_type=device, dtype=model_dtype):
        tokens, generated_text = sample(
            model=model,
            context=random_token,
            num_tokens=sampling_length,
            context_length=model_config.context_length,
            device=device,
            subset=subset
        )
    print("\n╭─ Generated Text Sample ──────────────")
    print("│")
    for line in generated_text.split('\n'):
        print(f"│ {line}")
    print("│")
    print("╰──────────────────────────────────────")
    writer.add_text("Sampled Text", generated_text, global_step)
    model.train()

def val_eval(
    model: CausalTransformer,
    writer: SummaryWriter,
    global_step: int,
    val_loader: DataLoader,
    device: str,
) -> None:
    model.eval()
    total_val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for val_batch in val_loader:
            x, y = val_batch
            x = x.to(device)
            y = y.to(device)
            model_dtype = next(model.parameters()).dtype
            with torch.autocast(device_type=device, dtype=model_dtype):
                logits = model(x)
                loss_value = F.cross_entropy(
                    logits.view(logits.size(0) * logits.size(1), logits.size(2)),
                    y.view(y.size(0) * y.size(1)),
                )
            loss_scalar = loss_value.item()
            total_val_loss += loss_scalar
            val_batches += 1
    if dist.get_rank() == 0:
        avg_val_loss = total_val_loss/val_batches
        writer.add_scalar("Loss/val", avg_val_loss, global_step)
        val_perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
        writer.add_scalar("Perplexity/val", val_perplexity, global_step)

        print("\n╭─ Validation Metrics ─────────────────")
        print(f"│ Step:                  {global_step}")
        print(f"│ Validation Loss:       {avg_val_loss:.5f}")
        print(f"│ Validation Perplexity: {val_perplexity:.5f}")
        print("╰──────────────────────────────────────")

    model.train()

def train(
    model: CausalTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    opt: optim.AdamW,
    device: str,
    training_config: TrainingConfig,
    model_config: ModelConfig,
    writer: SummaryWriter,
) -> None:
    global_step: int = 0

    steps_per_epoch = len(train_loader.dataset) // training_config.batch_size
    total_steps = steps_per_epoch * training_config.num_epochs

    lr_schedule = get_lr_schedule(
        schedule=training_config.lr_schedule,
        max_lr=training_config.learning_rate,
        warmup_steps=training_config.lr_warmup_steps,
        total_steps=total_steps
    )

    for e in range(training_config.num_epochs):
        train_loader.sampler.set_epoch(e)
        val_loader.sampler.set_epoch(e)

        if dist.get_rank() == 0:
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

            for param_group in opt.param_groups:
                param_group['lr'] = lr_schedule(global_step)

            if dist.get_rank() == 0:
                writer.add_scalar("Learning Rate", opt.param_groups[0]['lr'], global_step)

            model_dtype = next(model.parameters()).dtype
            with torch.autocast(device_type=device, dtype=model_dtype):
                logits: Logits = model(x)
                loss_value = F.cross_entropy(
                    logits.view(logits.size(0) * logits.size(1), logits.size(2)),
                    y.view(y.size(0) * y.size(1)),
                )

            loss_value.backward()
            opt.step()

            loss_scalar = loss_value.item()

            if dist.get_rank() == 0:
                writer.add_scalar("Loss/train", loss_scalar, global_step)
                writer.add_scalar(
                    "Perplexity/train",
                    torch.exp(torch.tensor(loss_scalar)).item(),
                    global_step
                )
                running_loss += loss_scalar

            if dist.get_rank() == 0:
                if global_step % training_config.reporting_steps == training_config.reporting_steps - 1:
                    last_loss = running_loss / training_config.reporting_steps
                    last_perplexity = torch.exp(torch.tensor(last_loss)).item()
                    print("\n╭─ Training Progress ──────────────────")
                    print(f"│ Batch Size: {training_config.batch_size}")
                    print(f"│ Step:       {i+1}")
                    print(f"│ Loss:       {last_loss:.5f}")
                    print(f"│ Perplexity: {last_perplexity:.5f}")
                    print("╰──────────────────────────────────────")
                running_loss = 0.0

            if global_step % training_config.validation_eval_steps == training_config.validation_eval_steps - 1:
                val_eval(
                    model=model,
                    writer=writer,
                    global_step=global_step,
                    val_loader=val_loader,
                    device=device
                )

            if dist.get_rank() == 0:
                if global_step % training_config.sampling_steps == training_config.sampling_steps - 1:
                    sample_and_log(
                        model=model,
                        global_step=global_step,
                        writer=writer,
                        device=device,
                        subset=train_loader.dataset,
                        model_config=model_config,
                        sampling_length=training_config.sampling_length_multiplier * model_config.context_length
                    )

            global_step += 1

if __name__ == '__main__':
    # default values are for toy model with shakespeare dataset
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

    setup_distributed()

    writer: SummaryWriter
    if dist.get_rank() == 0:
        writer = SummaryWriter()

    full_dataset: TiktokenTxtDataset = TiktokenTxtDataset(
        args.dataset, model_config.context_length
    )

    # this is mainly useful with char-level tokenizers
    assert full_dataset.get_vocab_size() == model_config.vocab_size, \
    f"dataset vocab size is {full_dataset.get_vocab_size()}, model_config vocab size is {model_config.vocab_size}"

    train_size = int(len(full_dataset) * (1-training_config.validation_split))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset=full_dataset,
        lengths=[train_size, val_size],
        generator=torch.Generator()
    )

    model = CausalTransformer(model_config=model_config)
    model = fsdp_wrap(model)

    device: str
    if training_config.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            torch.set_float32_matmul_precision("high")
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = training_config.device

    dtype = get_supported_dtype(device)
    # model = model.to(device, dtype=dtype)

    opt: optim.AdamW = optim.AdamW(model.parameters(), lr=training_config.learning_rate)

    if training_config.compile:
        if device == "mps":
            print("INFO: torch.compile not compatible with device mps, choosing not to compile")
        else:
            model = torch.compile(model)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        sampler=train_sampler
    )
    val_loader: DataLoader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        sampler=val_sampler
    )

    if dist.get_rank() == 0:
        trainable_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\n╭─ Training Info ──────────────────────")
        print(f"│ Total dataset size:   {len(full_dataset)}")
        print(f"│ Trainable parameters: {trainable_params}")
        print(f"│ Training set size:    {len(train_dataset)}")
        print(f"│ Training on device:   {device}")
        print(f"│ Validation set size:  {len(val_dataset)}")
        print(  "╰──────────────────────────────────────")

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        opt=opt,
        device=device,
        training_config=training_config,
        model_config=model_config,
        writer=writer,
    )

    if dist.get_rank() == 0:
        writer.flush()

    cleanup_distributed()

