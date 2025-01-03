# NOTE: training with placeholder dataset to begin with

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import CausalTransformer
from data import TxtFileDataset
from inference import sample
from utils import read_model_config, read_training_config

if __name__ == '__main__':

    model_config = read_model_config(r'config/small_model_config.json')
    training_config = read_training_config(r'config/small_training_config.json')

    # these just control logging
    REPORTING_FACTOR = 1000
    SAMPLING_FACTOR = 5000
    SAMPLING_LENGTH = 4 * model_config.context_length

    writer = SummaryWriter()

    dataset = TxtFileDataset(
        r"./datasets/tinyshakespeare.txt", model_config.context_length
    )
    assert dataset.get_vocab_size() == model_config.vocab_size

    model = CausalTransformer(model_config=model_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model has {trainable_params} trainable parameters")

    # determine the device we'll train on
    if training_config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_float32_matmul_precision("high")
    else:
        device = training_config.device
    model = model.to(device)
    print("running on device: ", device)

    opt = optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    scheduler = lr_scheduler.LinearLR(
        opt, 0.1, 1, training_config.lr_warmup_steps
    )  # lr increases every step

    model = torch.compile(model) if training_config.compile else model

    train_loader = DataLoader(
        dataset, batch_size=training_config.batch_size, shuffle=True
    )

    global_step = 0

    for e in range(training_config.num_epochs):
        print(f"epoch {e} —")
        running_loss = 0.0
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
            pseudo_batch_size = logits.size(0) * logits.size(1)
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

            if i % REPORTING_FACTOR == REPORTING_FACTOR - 1:
                last_loss = running_loss / REPORTING_FACTOR  # loss per batch
                print(
                    f"  batch size {training_config.batch_size}, step {i+1} loss: {last_loss:.5f}"
                )
                running_loss = 0.0

            if i % SAMPLING_FACTOR == SAMPLING_FACTOR - 1:
                model.eval()
                # batch size of 1, sequence length of 1 - just a random token
                random_token = torch.randint(
                    low=0, high=model_config.vocab_size, size=(1, 1)
                )
                # generate using sample method
                tokens, generated_text = sample(
                    model=model,
                    context=random_token,
                    num_tokens=SAMPLING_LENGTH,
                    context_length=model_config.context_length,
                    device=device,
                    dataset=dataset,
                )
                print("\nGenerated text sample:")
                print(generated_text)
                writer.add_text("Sampled Text", generated_text, global_step)
                print("\n")
                # return to training mode
                model.train()

    writer.flush()
