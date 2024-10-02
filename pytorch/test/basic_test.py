# placeholder tests for making sure forward passes are dimensionally correct
# also making sure they torch.compile

import torch
from personalpytorchtransformers.attention import MultiHeadAttention

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
    print(example_input)
    print(test_output)
    assert test_output.size() == example_input.size()


