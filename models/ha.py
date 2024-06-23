import torch
import ipdb

class HistoricalAverage:
    def __init__(self, seq_len=24, pred_len=24):
        self.maxlags = seq_len
        self.steps = pred_len

    def predict(self, x, y=None): # :month,day,weekday,hour
        x = x[..., :1]
        avg_x = torch.mean(x, 1)
        out = avg_x.unsqueeze(1).repeat(1, x.shape[1], 1, 1)
        return x
