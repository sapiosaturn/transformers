# placeholder tests for making sure forward passes are dimensionally correct
# also making sure they torch.compile

import torch
from personalpytorchtransformers.models import MultiHeadAttention, PositionWiseFeedForward, PositionalEncoding, DecoderBlock, DecoderOnlyTransformer

def test_mha():
    mha_module = MultiHeadAttention(
        embedding_dim=8,
        num_heads=2,
        context_length=4,
        attention_dropout_p=0,
        residual_dropout_p=0
    )
    mha_module = torch.compile(mha_module)
    # batch size of 5, seq len of 3, embedding dim of 8
    example_input = torch.randn((5, 3, 8))
    test_output = mha_module(example_input)
    assert test_output.size() == example_input.size()

def test_pwff():
    pwff_module = PositionWiseFeedForward(
        model_dim=8, 
        feedforward_dim=16
    )
    pwff_module = torch.compile(pwff_module)
    # batch size of 5, seq len of 3, embedding dim of 8
    example_input = torch.randn((5, 3, 8))
    test_output = pwff_module(example_input)
    assert test_output.size() == example_input.size()

def test_posenc():
    posenc_module = PositionalEncoding(
        context_length=4,
        model_dim=8
    )
    posenc_module = torch.compile(posenc_module)
    # batch size of 5, seq len of 3, embedding dim of 8
    example_input = torch.randn((5, 3, 8))
    test_output = posenc_module(example_input)
    assert test_output.size() == example_input.size()

def test_decoder_block():
    decoder_block_module = DecoderBlock(
        embedding_dim=8,
        num_heads=2,
        context_length=4,
        feedforward_dim=16,
        attention_dropout_p=0,
        residual_dropout_p=0
    )
    decoder_block_module = torch.compile(decoder_block_module)
    # batch size of 5, seq len of 3, embedding dim of 8
    example_input = torch.randn((5, 3, 8))
    test_output = decoder_block_module(example_input)
    assert test_output.size() == example_input.size()

def test_decoder_only_transformer():
    dec_only_transformer_module = DecoderOnlyTransformer(
        vocab_size=12,
        num_layers=3,
        embedding_dim=8,
        num_heads=2,
        context_length=4,
        feedforward_dim=16,
        attention_dropout_p=0,
        residual_dropout_p=0
    )
    dec_only_transformer_module = torch.compile(dec_only_transformer_module)
    # batch size of 5, seq len of 3
    example_input = torch.randint(low=0, high=12, size=(5, 3))
    test_output = dec_only_transformer_module(example_input)
    assert test_output.size() == torch.Size((5,3,12))



