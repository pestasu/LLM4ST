import torch
import torch.nn as nn
import numpy as np

class VARModel(nn.Module):
    def __init__(self, seq_len, pred_len):
        super(VARModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        input_dim = 1
        
        self.weights = nn.Parameter(torch.randn(input_dim, input_dim * seq_len))
        self.bias = nn.Parameter(torch.randn(input_dim))

    def forward(self, x):
        # x: [batch, num_station, seq_len, features]
        batch_size, num_station, seq_len, input_dim = x.shape
        
        x = x.permute(0, 1, 3, 2)  # [batch, num_station, features, seq_len]
        x = x.reshape(batch_size * num_station, input_dim * seq_len)  # [batch*num_station, features*seq_len]
        
        out = torch.matmul(x, self.weights.T) + self.bias  # [batch*num_station, features]
        out = out.view(batch_size, num_station, input_dim)  # [batch, num_station, features]
        return out
