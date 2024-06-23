import os
from math import sqrt
import ipdb
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer

from layers.embed import PatchEmbedding

import transformers
transformers.logging.set_verbosity_error()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class TimeLLM(nn.Module):
    def __init__(self, configs):
        super(TimeLLM, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 2
        self.d_llm = 4096
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        # 加载预训练的 Llama 配置 
        self.llama_config = LlamaConfig.from_pretrained('pretrained_model/meta-llama/Llama-2-7b-hf/')
        # self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
        self.llama_config.num_hidden_layers = configs.llm_layers # 设置 Llama 模型的隐藏层数
        self.llama_config.output_attentions = False # 输出注意力权重
        self.llama_config.output_hidden_states = False # 输出隐藏状态
        
        # 加载预训练的 Llama 模型
        self.llama = LlamaModel.from_pretrained(
            'pretrained_model/meta-llama/Llama-2-7b-hf/',
            # 'huggyllama/llama-7b',
            trust_remote_code=True,
            local_files_only=True, 
            config=self.llama_config,
            device_map={f'': configs.gpu}
        )

        # 加载预训练的 Llama 分词器
        self.tokenizer = LlamaTokenizer.from_pretrained(
            'pretrained_model/meta-llama/Llama-2-7b-hf/tokenizer.model',
            # 'huggyllama/llama-7b',
            trust_remote_code=True,
            local_files_only=True
        )

        # 如果分词器有 eos_token，则设置 pad_token 为 eos_token；否则，添加一个新的 pad_token
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token}) # 手动添加pad_token
            self.tokenizer.pad_token = pad_token

        # 冻结 Llama 模型的所有参数，不进行梯度更新
        for param in self.llama.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llama.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 100
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                head_dropout=configs.dropout)


        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]


    def forecast(self, x_enc=None, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x_enc = x_enc[..., :self.enc_in]
        B, T, C = x_enc.size()
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, C = x_enc.size()
        # x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * C, T, 1)

        min_values = torch.min(x_enc[..., :self.c_out], dim=1)[0]
        max_values = torch.max(x_enc[..., :self.c_out], dim=1)[0]
        medians = torch.median(x_enc[..., :self.c_out], dim=1).values
        lags = self.calcute_lags(x_enc[..., :self.c_out])
        trends = x_enc[..., :self.c_out].diff(dim=1).sum(dim=1) # 计算差分+求和
        prompt = [] # 构建prompt
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: Particulate matter 2.5 (PM2.5) prediction using air qulaity data and weather data."
                f"Task description: forecast the next {str(self.pred_len)} steps PM2.5 given the previous {str(self.seq_len)} steps air quality and weather data; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        # x_enc = x_enc.reshape(B, C, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids.to(x_enc.device) # prompt转换为pytorch张量，并填充和阶段->max_length
        
        prompt_embeddings = self.llama.get_input_embeddings()(prompt)  # (batch, prompt_token, dim) 嵌入向量 [B, L, d_llm]

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0) # 重新映射词嵌入 [num_tokens, d_llm]

        x_enc = x_enc.permute(0, 2, 1).contiguous() # B, C, T
        # ipdb.set_trace()
        # enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16)) # 补丁嵌入
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings) # 重新编程 B, P, d_llm
        prompt_embeddings2 = prompt_embeddings.unsqueeze(1).repeat(1, C, 1, 1).reshape(B * C, -1, self.d_llm) # B, L, d_llm
        ipdb.set_trace()
        llama_enc_out = torch.cat([prompt_embeddings2, enc_out], dim=1) # B, L+P, d_llm

        torch.cuda.empty_cache()
        llama_enc_out = llama_enc_out.to(torch.float16)
        dec_out = self.llama(inputs_embeds=llama_enc_out).last_hidden_state

        dec_out = dec_out[:, :, :self.d_ff] # B, L, D
        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1])) # B, C, L, D
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous() # B, C, D, L

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:]) # B, C, D, N
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')
        dec_out = dec_out[...,:self.c_out]
        return dec_out

    def calcute_lags(self, x_enc):
        # 傅里叶变换
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)

        # 计算自相关性的频域表示
        res = q_fft * torch.conj(k_fft)

        # 傅里叶逆变换
        corr = torch.fft.irfft(res, dim=-1)

        # 沿时间维度平均
        mean_value = torch.mean(corr, dim=1)

        # 找到自相关函数均值最大的前 self.top_k 个滞后值
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
