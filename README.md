# mirrorshift
personal repo for transformers code

- probably not well-organized or optimized, but getting there

## Contents

- `models.py`: Transformer implementations with GQA and MLA attention mechanisms, RoPE embeddings
- `train.py`: Training loop and logging logic
- `data.py`: Basic character-level tokenization and dataset handling
- `utils.py`: Configuration dataclasses and JSON loading
- `inference.py`: Text generation utilities
- `distributed.py`: Functions wrapping most of the distributed training related code
- `config/`: JSON files for model architecture and training parameters

## Usage

Clone the repository and `cd` into the folder.

Run `uv sync && source .venv/bin/activate`.

`cd` into the `mirrorshift` folder.

To train a tiny model on the coqa_stories dataset:
```bash
python train.py --model-config config/model_configs/small.json \
                --training-config config/training_configs/small.json \
                --dataset datasets/coqa_stories.txt
```
or just `python train.py`, since the above arguments are the default values.

To run on multiple GPUs:
```bash
torchrun --nproc_per_node=<number of GPUs> train.py
```
with any additional arguments (like above) as needed.

Visualize training with tensorboard (loss curves and text samples are saved):
```bash
tensorboard --logdir runs/
```
