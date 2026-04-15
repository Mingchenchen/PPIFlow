import torch
import torch.nn as nn


class SeqHead(nn.Module):
    def __init__(self, module_cfg):
        super(SeqHead, self).__init__()
        self._cfg = module_cfg
        self.input_dim = self._cfg.c_s
        self.hidden_dim = self._cfg.feat_dim
        self.layer_norm = nn.LayerNorm(self.input_dim)
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(self.hidden_dim, 21)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.output_layer(x)

        return x

