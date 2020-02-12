import torch
from torch import nn

class AttentionLayer(nn.Module):

    def __init__(self, input_size, n_ht):
        super().__init__()

        self.attn_mlp = nn.Linear(input_size, n_ht)
        self.attn_merge = nn.Linear( n_ht * 2 ,size_input)

    def forward(self, input, all_ht):

        self.attn_mlp()