import os 
import numpy as np
import ipdb
import torch
import torch.nn as nn
from math import sqrt

from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer

from utils.serialization import load_pkl
from layers.stformer import SFormer, TFormer
from layers.embed import STDataEmbedding

def sample_embed(embed, sample_size, start_idx, end_idx):
    embed_weight = embed.weight
    rand_idx = torch.randint(start_idx, end_idx, (sample_size,))
    return embed_weight[rand_idx].detach()

class ST4llm(nn.Module):
    def __init__(self, config):
        super(ST4llm, self).__init__()
        assert config.mode in ["pretrain", "predict"], "Error mode."
        self.config = config
        # self.device = config.gpu
        self.mode = config.mode
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.top_k = 2
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.prompt_tuning = config.prompt_tuning
        
        self.enc_embedding = STDataEmbedding(self.d_model)

        self.tformer = TFormer(config)
        self.sformer = SFormer(config)
        # load pre-trained st model
        if self.mode == 'predict':
            self.load_pre_trained_model()
        
        if config.backbone == 'gpt2':
            self.gpt2_config = GPT2Config.from_pretrained('pretrained_model/openai-community/gpt2/')
            self.gpt2_config.n_layer = config.llm_layers  
            self.gpt2_config.output_attentions = False 
            self.gpt2_config.output_hidden_states = False  

            self.backbone_llm = GPT2Model.from_pretrained(
                'pretrained_model/openai-community/gpt2/',
                trust_remote_code=True,
                local_files_only=True, 
                config=self.gpt2_config,
                # device_map={f'': self.device}
            )
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                'pretrained_model/openai-community/gpt2/',
                trust_remote_code=True,
                local_files_only=True
            )
            
            for i, (name, param) in enumerate(self.backbone_llm.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                elif 'mlp' in name and config.mlp == 1:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            self.d_llm = 768

        elif config.backbone == 'llama2':
            self.llama_config = LlamaConfig.from_pretrained('pretrained_model/meta-llama/Llama-2-7b-hf/')
            self.llama_config.num_hidden_layers = config.llm_layers 
            self.llama_config.output_attentions = False 
            self.llama_config.output_hidden_states = False 
        
            self.backbone_llm = LlamaModel.from_pretrained(
                'pretrained_model/meta-llama/Llama-2-7b-hf/',
                trust_remote_code=True,
                local_files_only=True, 
                config=self.llama_config,
                # device_map={f'': self.device}
            )

            self.tokenizer = LlamaTokenizer.from_pretrained(
                'pretrained_model/meta-llama/Llama-2-7b-hf/tokenizer.model',
                trust_remote_code=True,
                local_files_only=True
            )

            for param in self.backbone_llm.parameters():
                param.requires_grad = False
            
            self.d_llm = 4096

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})  
            self.tokenizer.pad_token = pad_token

        self.word_embeddings = self.backbone_llm.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 100
        self.mapping_t_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.mapping_s_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.aligning_t_layer = QFormerST(self.d_model, n_heads=2, d_keys=self.d_ff, d_llm=self.d_llm)
        self.aligning_s_layer = QFormerST(self.d_model, n_heads=2, d_keys=self.d_ff, d_llm=self.d_llm)

        self.patch_transform = nn.Conv2d(
                                self.d_llm,
                                self.d_llm,
                                kernel_size=(1, config.patch_size),
                                stride=(1, config.patch_size))

        self.output_projection = nn.Linear(self.d_ff*2, self.pred_len)

        if self.prompt_tuning:
            self.prompt_embedding = nn.Parameter(
                sample_embed(
                    embed=self.backbone_llm.get_input_embeddings(),
                    sample_size=config.num_prefix_emb,
                    start_idx=3,
                    end_idx=5003,
                )
            )

    def load_pre_trained_model(self, pretrained_path='final_model_0.pt'):
        """Load pre-trained model"""
        # load parameters
        checkpoint_root_path = f'pretrained_model/{self.config.model}/{self.config.data}'
        checkpoint_dict = torch.load(os.path.join(checkpoint_root_path, pretrained_path))
        state_dict_t = self.tformer.state_dict()
        state_dict_s = self.sformer.state_dict()

        new_state_dict_t, new_state_dict_s = {}, {}
        for k, v in state_dict_t.items():
            k = k.replace('module.', '')
            new_state_dict_t[k] = v
        state_dict_t.update(new_state_dict_t)
        self.tformer.load_state_dict(state_dict_t)

        for k, v in state_dict_s.items():
            k = k.replace('module.', '')
            new_state_dict_s[k] = v
        state_dict_s.update(new_state_dict_s)
        self.sformer.load_state_dict(state_dict_s)

        # freeze parameters
        for param in self.tformer.parameters():
            param.requires_grad = False
        for param in self.sformer.parameters():
            param.requires_grad = False

        print("***********The pretrain model is loaded..")
    
    def pretrain(self, long_history):
        """Feed forward of STDMAE.

        Args:
            long_history (torch.Tensor): Long-term historical data. shape: [B, L * P, N, 3]

        Returns:
            torch.Tensor: prediction with shape [B, N, L].
        """

        hidden_states_t, masked_tokens_t = self.tformer(long_history[..., [0]])
        hidden_states_s, masked_tokens_s = self.sformer(long_history[..., [0]])

        return hidden_states_t, masked_tokens_t, hidden_states_s, masked_tokens_s

    def predict(self, llm_input):
        llm_out = self.backbone_llm(inputs_embeds=llm_input).last_hidden_state
        llm_out = llm_out[..., :self.d_ff]
        return llm_out

    def forward(self, long_history):
        """Feed forward of STDMAE.
        Args:
            long_history (torch.Tensor): Long-term historical data. shape: [B, L * P, N, 3]

        Returns:
            torch.Tensor: prediction with shape [B, N, L].
        """
        if self.mode == 'pretrain':
            hidden_states_t, real_token_t, hidden_states_s, real_token_s = self.pretrain(long_history)
            return hidden_states_t, real_token_t, hidden_states_s, real_token_s

        else:
            long_term_history = long_history
            short_term_history = long_history[:, -self.seq_len:]
            B, T, N, C = short_term_history.size()

            if self.prompt_tuning:
                inputs_embeds = torch.cat(
                    [self.prompt_embedding[None, :, :].repeat((bs, 1, 1)), inputs_embeds], dim=1
                )  # [bs, prompt_len+max_seq_len, d_emb]
            prompt_embedding_t, source_embedding_t = self.get_manual_prompt_t(short_term_history) # [B, N, *, d_llm]
            prompt_embedding_s, source_embedding_s = self.get_manual_prompt_s(short_term_history) # [B, T, *, d_llm]
            
            short_term_history = short_term_history.contiguous()
            x_embed = self.enc_embedding(short_term_history)

            # extract pretrained ST features
            extracted_feature_t = self.tformer(long_term_history[..., [0]])
            extracted_feature_s = self.sformer(long_term_history[..., [0]])
            _, _, Pt, Dt = extracted_feature_t.shape
            _, _, Ps, Ds = extracted_feature_s.shape

            extracted_feature_t = extracted_feature_t.reshape(B*N, Pt, Dt)
            extracted_feature_s = extracted_feature_s.permute(0, 2, 1, 3).reshape(B*Ps, N, Ds)
            aligned_feature_t = self.aligning_t_layer(extracted_feature_t, source_embedding_t, source_embedding_t) # [B*N, Pt, d_llm]
            aligned_feature_s = self.aligning_s_layer(extracted_feature_s, source_embedding_s, source_embedding_s) # [B*Ps, N, d_llm]

            aligned_feature_t = aligned_feature_t.reshape(B, N, Pt, -1)[:,:,-1:] # [B, N, Pt, d_llm]
            aligned_feature_s = aligned_feature_s.reshape(B, Ps, N, -1)[:,-1:] # [B, Ps, N, d_llm]
            prompt_embedding_s = self.patch_transform(prompt_embedding_s.permute(0,3,2,1)).permute(0,3,2,1)

            llm_inp_t = torch.cat([prompt_embedding_t, aligned_feature_t], dim=2).reshape(B*N, -1, self.d_llm) # [B, N, Lt+Pt, d_llm]
            llm_inp_s = torch.cat([prompt_embedding_s, aligned_feature_s], dim=2).reshape(B, -1, self.d_llm) # [B, T, Ls+N, d_llm]
            dec_out_t = self.predict(llm_inp_t).reshape(B, N, -1, self.d_ff) # B, N, Lt+Pt, D
            dec_out_s = self.predict(llm_inp_s).reshape(B, 1, -1, self.d_ff) # B, 1, Ls+N, D

            dec_out_t = dec_out_t[:, :, -1:].squeeze(-2) # B, N, D
            dec_out_s = dec_out_s[:, :, -N:].squeeze(1) # B, T, N, D
            dec_out = torch.cat([dec_out_t, dec_out_s], -1)
            dec_out = self.output_projection(dec_out)
            dec_out = dec_out.unsqueeze(-1).transpose(1, 2)
        
            return dec_out
    
    def get_manual_prompt_t(self, short_term_history):
        short_history = short_term_history[...,:1]
        B, T, N, C = short_history.size()

        # time features
        min_values_t = torch.min(short_history, dim=1)[0]
        max_values_t = torch.max(short_history, dim=1)[0]
        medians_t = torch.median(short_history, dim=1).values
        trends = short_history.diff(dim=1).sum(dim=1) 
        adj_mx = load_pkl(os.path.join(self.config.data_root_path, 'adj_mx.pkl'))
        if 'AIR' in self.config.data:
            adj_mx = adj_mx[2]

        prompts = []
        for b in range(B):
            for n in range(N):
                min_t_str = str(min_values_t[b, n].tolist())
                max_t_str = str(max_values_t[b, n].tolist())
                median_t_str = str(medians_t[b, n].tolist())
                trends_str = 'upward' if trends[b, n] > 0 else 'downward'
                adj_ids = np.nonzero(adj_mx[n])[0].tolist()
                adjs_str = ', '.join([f'{idx+1}th' for idx in adj_ids])
                prompt = (
                f"<|start_prompt|>Dataset description: Traffic flow prediction using time features."
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps traffic flow; "
                "Input statistics: "
                f"min value {min_t_str}, "
                f"max value {max_t_str}, "
                f"median value {median_t_str}, "
                f"the trend of input is {trends_str}, "
                f"The {n+1}th node is related to the {adjs_str} nodes."
                )
                prompts.append(prompt)

        input_prompt = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids.to(short_history.device)

        prompt_embedding_t = self.backbone_llm.get_input_embeddings()(input_prompt).reshape(B, N, input_prompt.shape[-1], -1) # [B, N, d, d_llm]

        source_embedding_t = self.mapping_t_layer(self.word_embeddings.permute(1, 0)).permute(1, 0) # [n_token, d_llm]

        return prompt_embedding_t, source_embedding_t

    def get_manual_prompt_s(self, short_term_history):
        short_history = short_term_history[...,:1]
        B, T, N, C = short_history.size()

        # space features
        min_values_s = torch.min(short_history, dim=2)[0]
        max_values_s = torch.max(short_history, dim=2)[0]
        medians_s = torch.median(short_history, dim=2).values

        prompts = []
        for b in range(B):
            for t in range(T):
                min_s_str = str(min_values_s[b, t].tolist())
                max_s_str = str(max_values_s[b, t].tolist())
                median_s_str = str(medians_s[b, t].tolist())
                day_of_week = short_term_history[b,t,0,1]
                hour_of_day = short_term_history[b,t,0,2]
                prompt = (
                f"<|start_prompt|>Dataset description: Traffic flow prediction using space features."
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps traffic flow; "
                "Input statistics: "
                f"min value {min_s_str}, "
                f"max value {max_s_str}, "
                f"median value {median_s_str},"
                f"The day of the week is {day_of_week}, and the hour of the day is {hour_of_day}."
                )
                prompts.append(prompt)

        input_prompt = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids.to(short_history.device)

        prompt_embedding_s = self.backbone_llm.get_input_embeddings()(input_prompt).reshape(B, T, input_prompt.shape[-1], -1) # [B, T, d, d_llm]

        source_embedding_s = self.mapping_s_layer(self.word_embeddings.permute(1, 0)).permute(1, 0) # [n_token, d_llm]

        return prompt_embedding_s, source_embedding_s


class QFormerST(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(QFormerST, self).__init__()
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

        scale = 1. / sqrt(target_embedding.shape[-1])

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        reprogramming_embedding = reprogramming_embedding.reshape(B, L, -1)

        return self.out_projection(reprogramming_embedding)