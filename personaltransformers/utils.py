import json
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    num_layers: int
    num_kv_heads: int
    embedding_dim: int
    num_heads: int
    context_length: int
    feedforward_dim: int
    attention_dropout_p: float
    residual_dropout_p: float
    attention_type: Literal["gqa", "mla"]
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    v_head_dim: int | None = None

@dataclass(frozen=True)
class TrainingConfig:
    device: str
    batch_size: int
    learning_rate: float
    lr_warmup_steps: int
    num_epochs: int
    compile: bool

def read_training_config(config_path: str) -> TrainingConfig:
    with open(config_path, 'r') as file:
        config_data = json.load(file)
        required_keys = {'device', 'batch_size', 'learning_rate', 'lr_warmup_steps', 'num_epochs', 'compile'}
        missing_keys = required_keys - config_data.keys()
        if missing_keys:
            raise ValueError(f"Missing configuration options in {config_path}: {', '.join(missing_keys)}")
        return TrainingConfig(
            device=config_data['device'],
            batch_size=config_data['batch_size'],
            learning_rate=config_data['learning_rate'],
            lr_warmup_steps=config_data['lr_warmup_steps'],
            num_epochs=config_data['num_epochs'],
            compile=config_data['compile']
        )

def read_model_config(config_path: str) -> ModelConfig:
    with open(config_path, 'r') as file:
        config_data = json.load(file)
        required_keys = {
            'vocab_size', 'num_layers', 'num_kv_heads', 'embedding_dim',
            'num_heads', 'context_length', 'feedforward_dim',
            'attention_dropout_p', 'residual_dropout_p', 'attention_type'
        }
        missing_keys = required_keys - config_data.keys()
        if missing_keys:
            raise ValueError(f"Missing configuration options in {config_path}: {', '.join(missing_keys)}")
        return ModelConfig(
            vocab_size=config_data['vocab_size'],
            num_layers=config_data['num_layers'],
            num_kv_heads=config_data['num_kv_heads'],
            embedding_dim=config_data['embedding_dim'],
            num_heads=config_data['num_heads'],
            context_length=config_data['context_length'],
            feedforward_dim=config_data['feedforward_dim'],
            attention_dropout_p=config_data['attention_dropout_p'],
            residual_dropout_p=config_data['residual_dropout_p'],
            attention_type=config_data['attention_type'],
            q_lora_rank=config_data.get('q_lora_rank'),
            kv_lora_rank=config_data.get('kv_lora_rank'),
            qk_nope_head_dim=config_data.get('qk_nope_head_dim'),
            qk_rope_head_dim=config_data.get('qk_rope_head_dim'),
            v_head_dim=config_data.get('v_head_dim')
        )