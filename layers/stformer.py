import ipdb
import math
import random
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from timm.models.vision_transformer import trunc_normal_
from .embed import PatchEmbedding_2d, PositionalEncoding_2d



def unshuffle(shuffled_tokens):
    dic = {}
    for k, v, in enumerate(shuffled_tokens):
        dic[v] = k
    unshuffle_index = []
    for i in range(len(shuffled_tokens)):
        unshuffle_index.append(dic[i])
    return unshuffle_index

class MaskGenerator(nn.Module):
    """Mask generator."""
    def __init__(self, mask_ratio):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.sort = True

    def forward(self, num_tokens):
        mask = list(range(int(num_tokens))) # [0,1,...]
        random.shuffle(mask)
        mask_len = int(num_tokens * self.mask_ratio)
        masked_tokens = mask[:mask_len]
        unmasked_tokens = mask[mask_len:]
        if self.sort:
            masked_tokens = sorted(masked_tokens)
            unmasked_tokens = sorted(unmasked_tokens)
        return unmasked_tokens, masked_tokens

class TransformerLayers(nn.Module):
    def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*mlp_ratio, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src):
        B, N, L, D = src.shape
        src = src * math.sqrt(self.d_model)
        src=src.contiguous()
        src = src.view(B*N, L, D)
        src = src.transpose(0, 1)
        output = self.transformer_encoder(src, mask=None)
        output = output.transpose(0, 1).view(B, N, L, D)
        return output

class SFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.mode in ["pretrain", "predict"], "Error mode."
        self.mode = config.mode
        self.patch_size = config.patch_size
        self.in_channel = config.feat_dims[0]
        self.embed_dim = config.d_model
        self.num_heads = config.num_heads
        self.mask_ratio = config.mask_ratio
        self.encoder_layer = config.encoder_layer
        self.decoder_layer = config.decoder_layer
        self.mlp_layer = config.mlp_layer
        self.spatial = config.spatial
        self.dropout = config.dropout
        self.selected_feature = 0

        # norm layers
        self.pos_mat=None

        # # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.embed_dim))
        self.maskg = MaskGenerator(self.mask_ratio) # uniform_rand generate mask

        # encoder specifics
        # # patchify & embedding
        self.patch_embedding = PatchEmbedding_2d(self.patch_size, self.in_channel, self.embed_dim, norm_layer=None)
        # # positional encoding
        self.positional_encoding = PositionalEncoding_2d()

        # encoder
        self.encoder = TransformerLayers(self.embed_dim, self.encoder_layer, self.mlp_layer, self.num_heads, self.dropout)
        self.encoder_norm = nn.LayerNorm(self.embed_dim) # norm layers

        # decoder 
        self.enc2dec_emb = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.decoder = TransformerLayers(self.embed_dim, self.decoder_layer, self.mlp_layer, self.num_heads, self.dropout)
        self.decoder_norm = nn.LayerNorm(self.embed_dim)

        # prediction (reconstruction) layer
        self.output_layer = nn.Linear(self.embed_dim, self.patch_size)
        self.initialize_weights()

    def initialize_weights(self):
        trunc_normal_(self.mask_token, std=.02)

    def encoding(self, long_term_history, mask=True):
        """

        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, N, C, P * L],
                                                which is used in the Pre-training.
                                                P is the number of patches.
            mask (bool): True in pre-training stage and False in forecasting stage.

        Returns:
            torch.Tensor: hidden states of unmasked tokens
            list: unmasked token index
            list: masked token index
        """

        # patchify and embed input
        # patching embedding
        patches = self.patch_embedding(long_term_history)  # B, N, d, P
        patches = patches.transpose(-1, -2)  # B, N, P, d

        # positional embedding
        patches, self.pos_mat = self.positional_encoding(patches)      
        batch_size, num_nodes, num_patches, num_dim = patches.shape  

        if mask:
            unmasked_token_index, masked_token_index = self.maskg(patches.shape[1])
            encoder_input = patches[:, unmasked_token_index, :, :]
        else:
            unmasked_token_index, masked_token_index = None, None
            encoder_input = patches# B, N, P, d

        encoder_input = encoder_input.transpose(1, 2) # B, P, N, d
        hidden_states_unmasked = self.encoder(encoder_input) # B, P, N, d
        hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, -1, num_patches, self.embed_dim) # B, N, P, d


        return hidden_states_unmasked, unmasked_token_index, masked_token_index

    def decoding(self, hidden_states_unmasked, masked_token_index, unmasked_token_index):

        # encoder 2 decoder layer
        hidden_states_unmasked = self.enc2dec_emb(hidden_states_unmasked) # B, N*, P, d
        # B,N*r,P,d
        batch_size, num_nodes, num_patches, _ = hidden_states_unmasked.shape 
        # unmasked_token_index=[i for i in range(0,len(masked_token_index)+num_nodes) if i not in masked_token_index ]
        hidden_states_masked = self.pos_mat[:,masked_token_index] # B, N*r,P, d
        hidden_states_masked += self.mask_token.expand(batch_size, len(masked_token_index), num_patches, hidden_states_unmasked.shape[-1]) # B, N*r,P,  d
        hidden_states_unmasked += self.pos_mat[:,unmasked_token_index] # ,N*(1-r), P, d

        # hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=1)   # B, P, N, d
        # concatanate according to the sequence order
        hidden_states_full = torch.zeros(batch_size, num_nodes + len(masked_token_index), num_patches, hidden_states_unmasked.shape[-1], device=hidden_states_unmasked.device)
        hidden_states_full[:, masked_token_index] = hidden_states_masked
        hidden_states_full[:, unmasked_token_index] = hidden_states_unmasked

        # decoding
        hidden_states_full = hidden_states_full.transpose(2, 1)
        hidden_states_full = self.decoder(hidden_states_full) # B, P, N, d
        hidden_states_full = self.decoder_norm(hidden_states_full)# B, P, N, d

        # prediction (reconstruction)
        hidden_states_full = hidden_states_full.transpose(2, 1)
        reconstruction_full = self.output_layer(hidden_states_full) # B, P, N, L

        return reconstruction_full

    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, unmasked_token_index,
                                        masked_token_index):
        """Get reconstructed masked tokens and corresponding ground-truth for subsequent loss computing.

        Args:
            reconstruction_full (torch.Tensor): reconstructed full tokens.
            real_value_full (torch.Tensor): ground truth full tokens.
            unmasked_token_index (list): unmasked token index.
            masked_token_index (list): masked token index.

        Returns:
            torch.Tensor: reconstructed masked tokens.
            torch.Tensor: ground truth masked tokens.
        """
        # get reconstructed masked tokens
        batch_size, num_nodes, num_patches, _ = reconstruction_full.shape # B, N, P, L
        reconstruction_masked_tokens = reconstruction_full[:, masked_token_index]     # B, r*N, P, L
        reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, len(masked_token_index), -1)  # B, r*N, P*L

        label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size).transpose(1, 2).squeeze(-2)  # B, N, P, L
        label_masked_tokens = label_full[:, masked_token_index].contiguous() # B, N*r, P, L
        label_masked_tokens = label_masked_tokens.view(batch_size, len(masked_token_index), -1)  # B, r*N, P*L

        return reconstruction_masked_tokens, label_masked_tokens

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, epoch: int = None, **kwargs) -> torch.Tensor:
        # reshape
        history_data = history_data.permute(0, 2, 3, 1)     # B, N, 1, L * P
        
        # feed forward
        if self.mode == "pretrain":
            # encoding
            hidden_states_unmasked, unmasked_token_index, masked_token_index = self.encoding(history_data)
            # decoding
            reconstruction_full = self.decoding(hidden_states_unmasked, masked_token_index, unmasked_token_index)
            # for subsequent loss computing
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(reconstruction_full, history_data, unmasked_token_index, masked_token_index)

            return reconstruction_masked_tokens, label_masked_tokens
        else:
            hidden_states_full, _, _ = self.encoding(history_data, mask=False)

            return hidden_states_full

class TFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.mode in ["pretrain", "predict"], "Error mode."
        self.mode = config.mode
        self.patch_size = config.patch_size
        self.in_channel = config.feat_dims[0]
        self.embed_dim = config.d_model
        self.num_heads = config.num_heads
        self.mask_ratio = config.mask_ratio
        self.encoder_layer = config.encoder_layer
        self.decoder_layer = config.decoder_layer
        self.mlp_layer = config.mlp_layer
        self.dropout = config.dropout
        self.selected_feature = 0

        # # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.embed_dim))
        self.maskg = MaskGenerator(self.mask_ratio) # uniform_rand generate mask

        # encoder specifics
        # # patchify & embedding
        self.patch_embedding = PatchEmbedding_2d(self.patch_size, self.in_channel, self.embed_dim, norm_layer=None)
        # # positional encoding
        self.positional_encoding = PositionalEncoding_2d()
        self.pos_mat=None 

        # encoder
        self.encoder = TransformerLayers(self.embed_dim, self.encoder_layer, self.mlp_layer, self.num_heads, self.dropout)
        self.encoder_norm = nn.LayerNorm(self.embed_dim) # norm layers
        # decoder
        self.enc2dec_emb = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.decoder = TransformerLayers(self.embed_dim, self.decoder_layer, self.mlp_layer, self.num_heads, self.dropout)
        self.decoder_norm = nn.LayerNorm(self.embed_dim) # norm layers

        # prediction (reconstruction) layer
        self.output_layer = nn.Linear(self.embed_dim, self.patch_size)
        self.initialize_weights()

    def initialize_weights(self):
        trunc_normal_(self.mask_token, std=.02)

    def encoding(self, long_term_history, mask=True):
        """

        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, N, 1, P * L],
                                                which is used in the Pre-training.
                                                P is the number of patches.
            mask (bool): True in pre-training stage and False in forecasting stage.

        Returns:
            torch.Tensor: hidden states of unmasked tokens
            list: unmasked token index
            list: masked token index
        """

        # patchify and embed input
        # patching embedding
        patches = self.patch_embedding(long_term_history)  # B, N, d, P
        patches = patches.transpose(-1, -2)  # B, N, P, d

        # positional embedding
        patches, self.pos_mat = self.positional_encoding(patches)    
        batch_size, num_nodes, num_patches, num_dim = patches.shape

        if mask:
            # uniform mask
            unmasked_token_index, masked_token_index = self.maskg(num_patches)
            encoder_input = patches[:, :, unmasked_token_index, :]
        else:
            unmasked_token_index, masked_token_index = None, None
            encoder_input = patches 
        
        # encoder
        hidden_states_unmasked = self.encoder(encoder_input) # B, N, P, d
        hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1, self.embed_dim)# B, N, P, d

        return hidden_states_unmasked, unmasked_token_index, masked_token_index

    def decoding(self, hidden_states_unmasked, masked_token_index, unmasked_token_index):

        # encoder 2 decoder layer
        hidden_states_unmasked = self.enc2dec_emb(hidden_states_unmasked) # B, N, P*, d
        batch_size, num_nodes, num_patches, _ = hidden_states_unmasked.shape
        # unmasked_token_index = [i for i in range(0,len(masked_token_index) + num_patches) if i not in masked_token_index]
        hidden_states_masked = self.pos_mat[:,:,masked_token_index,:] # B, N, Pm, d
        hidden_states_masked += self.mask_token.expand(batch_size, num_nodes, len(masked_token_index), hidden_states_unmasked.shape[-1])
        hidden_states_unmasked += self.pos_mat[:,:,unmasked_token_index,:]

        # hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)   # B, N, P, d
        # concatanate according to the sequence order
        hidden_states_full = torch.zeros(batch_size, num_nodes, num_patches + len(masked_token_index), hidden_states_unmasked.shape[-1], device=hidden_states_unmasked.device)
        hidden_states_full[:,:,masked_token_index] = hidden_states_masked
        hidden_states_full[:,:,unmasked_token_index] = hidden_states_unmasked

        # decoding
        hidden_states_full = self.decoder(hidden_states_full)
        hidden_states_full = self.decoder_norm(hidden_states_full)

        # prediction (reconstruction)
        reconstruction_full = self.output_layer(hidden_states_full.view(batch_size, num_nodes, -1, self.embed_dim))

        return reconstruction_full

    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, unmasked_token_index,
                                        masked_token_index):
        """Get reconstructed masked tokens and corresponding ground-truth for subsequent loss computing.

        Args:
            reconstruction_full (torch.Tensor): reconstructed full tokens.
            real_value_full (torch.Tensor): ground truth full tokens.
            unmasked_token_index (list): unmasked token index.
            masked_token_index (list): masked token index.

        Returns:
            torch.Tensor: reconstructed masked tokens.
            torch.Tensor: ground truth masked tokens.
        """
        # get reconstructed masked tokens
        batch_size, num_nodes, num_patches, _ = reconstruction_full.shape

        reconstruction_masked_tokens = reconstruction_full[:, :, masked_token_index, :]     # B, N, r*P, d
        reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2) # B, r*P*d, N

        label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size).transpose(1, 2).squeeze(-2) # B, N, P, L
        label_masked_tokens = label_full[:, :, masked_token_index, :].contiguous() # B, N, r*P, d
        label_masked_tokens = label_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)  # B, r*P*d, N

        return reconstruction_masked_tokens, label_masked_tokens

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, epoch: int = None, **kwargs) -> torch.Tensor:
        """Feed forward of TFormer.

        Args:
            x (torch.Tensor): Long-term historical data. shape: [B, L * P, N, 1]

        Returns:
            torch.Tensor: prediction with shape [B, N, L].
        """
        # reshape
        history_data = history_data.permute(0, 2, 3, 1)     # B, N, 1, L * P
        
        # feed forward
        if self.mode == "pretrain":
            # encoding
            hidden_states_unmasked, unmasked_token_index, masked_token_index = self.encoding(history_data)
            # decoding
            reconstruction_full = self.decoding(hidden_states_unmasked, masked_token_index, unmasked_token_index)
            # for subsequent loss computing
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(reconstruction_full, history_data, unmasked_token_index, masked_token_index)
            return reconstruction_masked_tokens, label_masked_tokens

        else:
            hidden_states_full, _, _ = self.encoding(history_data, mask=False)
            return hidden_states_full
