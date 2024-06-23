import math
import ipdb
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import RoIPool
from torch.nn.utils import weight_norm
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer

class STDataEmbedding(nn.Module):
    def __init__(self, d_model, kernel_size=2, dropout=0.1):
        super(STDataEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.value_embedding = nn.Conv2d(in_channels=1, out_channels=d_model,
                                   kernel_size=(1, kernel_size), padding=padding, padding_mode='circular', bias=False)
        # self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.ln_day = nn.Linear(1, d_model)
        self.ln_week = nn.Linear(1, d_model)
        self.ln1 = nn.Linear(d_model, d_model)
        self.ln2 = nn.Linear(d_model, d_model)
        self.ln = nn.Linear(d_model, d_model)
        self.act = nn.ReLU()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x[..., :1].permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        day = self.ln_day(x[..., 1:2])
        week = self.ln_week(x[..., 2:3])
        time_embed = self.ln(self.act(self.ln2(self.act(self.ln1(day + week)))))

        return x + self.dropout(time_embed)


class PositionalEncoding_2d(nn.Module):
    """Positional encoding."""

    def __init__(self):
        super().__init__()


    def forward(self, input_data, index=None, abs_idx=None):
        """Positional encoding

        Args:
            input_data (torch.tensor): input sequence with shape [B, N, P, d].
            index (list or None): add positional embedding by index.

        Returns:
            torch.tensor: output sequence
        """

        batch_size, num_nodes, num_patches, num_feat = input_data.shape
        tp_enc_2d = PositionalEncoding2D(num_feat)
        input_data+=tp_enc_2d(input_data)
        return input_data,tp_enc_2d(input_data)

class PatchEmbedding_2d(nn.Module):
    """Patchify time series."""

    def __init__(self, patch_size, in_channel, embed_dim, norm_layer):
        super().__init__()
        self.output_channel = embed_dim
        self.len_patch = patch_size             # the L
        self.input_channel = in_channel
        self.output_channel = embed_dim
        self.input_embedding = nn.Conv2d(
                                        in_channel,
                                        embed_dim,
                                        kernel_size=(self.len_patch, 1),
                                        stride=(self.len_patch, 1))
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()

    def forward(self, long_term_history):
        """
        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, N, 1, P * L],
                                                which is used in the TSFormer.
                                                P is the number of segments (patches).

        Returns:
            torch.Tensor: patchified time series with shape [B, N, d, P]
        """

        batch_size, num_nodes, num_feat, len_time_series = long_term_history.shape
        long_term_history = long_term_history.unsqueeze(-1) # B, N, C, L, 1
        # B*N,  C, L, 1
        long_term_history = long_term_history.reshape(batch_size*num_nodes, num_feat, len_time_series, 1)
        # B*N,  d, L/P, 1
        output = self.input_embedding(long_term_history)
        # norm
        output = self.norm_layer(output)
        # reshape
        output = output.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1)    # B, N, d, P
        assert output.shape[-1] == len_time_series / self.len_patch
        return output

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = x.to(self.tokenConv.weight.dtype)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)

class ReplicationPad1d(nn.Module):
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([input, replicate_padding], dim=-1)
        return output

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        # self.padding_patch_layer = ReplicationPad1d((0, stride))
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # Positional embedding
        # self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) #+ self.position_embedding(x)
        return self.dropout(x), n_vars

class DataEmbedding_wo_time(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_time, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

