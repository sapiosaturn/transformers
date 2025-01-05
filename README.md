# transformers
personal repo for transformers code

- probably not well-organized or optimized, but getting there

## Contents

- `models.py`: Transformer implementations with GQA and MLA attention mechanisms, RoPE embeddings
- `train.py`: Training loop and logging logic
- `data.py`: Basic character-level tokenization and dataset handling
- `utils.py`: Configuration dataclasses and JSON loading
- `inference.py`: Text generation utilities
- `config/`: JSON files for model architecture and training parameters

## Usage

To train on the tiny shakespeare dataset:
```python
python train.py --model-config config/model_configs/ss_small.json \
                --training-config config/training_configs/ss_small.json \
                --dataset ./datasets/tinyshakespeare.txt
```

Visualize training with tensorboard (loss curves and text samples are saved):
```bash
tensorboard --logdir runs/
```
