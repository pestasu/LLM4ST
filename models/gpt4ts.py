from typing import Optional
import numpy as np
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from transformers import GPT2ForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel
from einops import rearrange
from layers.embed import DataEmbedding, DataEmbedding_wo_time


class GPT4ts(nn.Module):
    
    def __init__(self, configs):
        super(GPT4ts, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.patch_num = (configs.seq_len + self.pred_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.enc_embedding = DataEmbedding(c_in=configs.enc_in * self.patch_size, d_model=configs.d_model, dropout=configs.dropout)

        # self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2 = GPT2Model.from_pretrained('pretrained_model/openai-community/gpt2/', local_files_only=True)

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name: # or 'mlp' in name:
                param.requires_grad = True
            elif 'mlp' in name and configs.mlp == 1:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # self.in_layer = nn.Linear(configs.patch_size, configs.d_model)

        self.predict_linear_pre = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.predict_linear = nn.Linear(self.patch_size, configs.enc_in)
        self.ln = nn.LayerNorm(configs.d_ff)
        self.out_layer = nn.Linear(configs.d_ff, configs.c_out)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x_enc = x_enc[..., :self.enc_in]
        # ipdb.set_trace()
        B, T, C = x_enc.size()
        
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear_pre(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        enc_out = torch.nn.functional.pad(enc_out, (0, 768-enc_out.shape[-1]))

        # enc_out = rearrange(enc_out, 'b l m -> b m l')
        # enc_out = self.padding_patch_layer(enc_out)
        # enc_out = enc_out.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # enc_out = self.predict_linear(enc_out)
        # enc_out = rearrange(enc_out, 'b m n p -> b n (m p)')
        
        dec_out = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]
        # dec_out = dec_out.reshape(B, -1)
        
        # dec_out = self.ln(dec_out)
        dec_out = self.out_layer(dec_out)
        # print(dec_out.shape)
        # dec_out = dec_out.reshape(B, self.pred_len + self.seq_len, -1)
        
        # De-Normalization from Non-stationary Transformer
        
        dec_out = dec_out * \
                    (stdev[:, 0, :self.c_out].unsqueeze(1).repeat(
                        1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :self.c_out].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        
        return dec_out


    
