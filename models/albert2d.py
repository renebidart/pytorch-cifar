'''https://github.com/graykode/ALBERT-Pytorch
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_




def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class LayerNormCustom(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, n_hidden, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_hidden))
        self.beta  = nn.Parameter(torch.zeros(n_hidden))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class PositionEmbedding(nn.Module):
    "Position embeddings"
    def __init__(self, n_hidden, H, W, drop=.1):
        super().__init__()
        self.pos_embed_h = nn.Embedding(H, n_hidden) # position embedding
        self.pos_embed_w = nn.Embedding(W, n_hidden) # position embedding
        self.embedding = nn.Conv2d(n_hidden, n_hidden, kernel_size=1, bias=True)
        # !!!! ???? maybe switch to good norm
        self.norm = nn.BatchNorm2d(n_hidden)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        BS, D, H, W = x.size()
        seq_len = H
        pos_h = torch.arange(H, dtype=torch.long, device=x.device)
        pos_w = torch.arange(W, dtype=torch.long, device=x.device)
        # H -> [BS, H, W]
        pos_h = pos_h.unsqueeze(0).unsqueeze(-1).repeat(BS, 1, W)
        pos_w = pos_w.unsqueeze(0).unsqueeze(0).repeat(BS, H, 1)
#         print('pos_h[0, 0, :]', pos_h[0, 0, :])
#         print('pos_w[0, 0, :]', pos_w[0, 0, :])
#         print('pos_w[0, :, 2]', pos_w[0, :, 2])
        # not sure if it should be residual like this:
        pos_h = self.pos_embed_h(pos_h).permute(0, 3, 1, 2).contiguous()
        pos_w =  self.pos_embed_w(pos_w).permute(0, 3, 1, 2).contiguous()
        x = x + self.embedding(x) + pos_h + pos_w
        return self.drop(self.norm(x))

## Copy paste from https://github.com/leaderj1001/Stand-Alone-Self-Attention/blob/a983f0f643632b1f2b7b8b27693182f22e9e574c/attention.py#L9
## !!!!!! ???? Check if this implementation is right, or try something different like non-local block from
# https://github.com/KaiyuYue/cgnl-network.pytorch/blob/master/model/resnet.py
# class AttentionConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
#         super(AttentionConv, self).__init__()
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.groups = groups
#         assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

#         self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
#         self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)
#         self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
#         self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
#         self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
#         self.reset_parameters()
        
#     def forward(self, x):
#         batch, channels, height, width = x.size()
#         padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
#         q_out = self.query_conv(x)
#         k_out = self.key_conv(padded_x)
#         v_out = self.value_conv(padded_x)

# #         k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
#         v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
#         k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
#         k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)
#         ## !!!! below is added based on comment
#         k_out = F.unfold(k_out, kernel_size=(self.kernel_size, self.kernel_size), stride=self.stride)
#         k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
#         v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

#         q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)
#         out = (q_out * k_out).sum(dim=2)
#         out = F.softmax(out, dim=-1)
#         out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)
#         return out

#     def reset_parameters(self):
#         init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
#         init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
#         init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
#         init.normal_(self.rel_h, 0, 1)
#         init.normal_(self.rel_w, 0, 1)      
        
        
class MultiheadSelfAttention2D(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, need_weights=False):
        super(MultiheadSelfAttention2D, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        
        self.dropout = dropout
        self.need_weights = need_weights
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
 
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()
        
    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

        
    def forward(self, src):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r""" No weights / masks, this is for imgs, and forced to do only self attention
    Args:
        src: this is query, key, and value. 
        need_weights: output attn_output_weights.

    Shape:
        - Inputs:
        - src [BS, D, H, W] (img format), we convert this to NLP style to use multihead attn

        - Outputs:
        - attn_output: :math:`(BS, D, H, W)` 
        - attn_output_weights: :math:`(BS, H*W, H*W)` 
        """

        # NLP likes format: [L, BS, D]. Convert our [BS, D, H, W] -> [H*W, BS, D]:
        BS, D, H, W = src.size()
        src = src.view(BS, D, H*W)
        src = src.permute(2, 0, 1).contiguous()
        
        # this is self attention, so all the same
        query, key, value = src, src, src
        
        out, attn_weights = F.multi_head_attention_forward(
                                    query, key, value, 
                                    embed_dim_to_check=self.embed_dim,
                                    num_heads=self.num_heads,
                                    in_proj_weight = self.in_proj_weight, 
                                    in_proj_bias = self.in_proj_bias,
                                    bias_k=None,  
                                    bias_v=None,
                                    add_zero_attn=False, # ??? don't understand
                                    dropout_p=self.dropout, 
                                    out_proj_weight=self.out_proj.weight,
                                    out_proj_bias=self.out_proj.bias,
                                    training=self.training,
                                    key_padding_mask=None, 
                                    need_weights=self.need_weights,
                                    attn_mask=None)
        
        # [H*W, BS, D] -> [BS, D, H*W]:
        out = out.permute(1, 2, 0).contiguous()
        out = out.view(BS, D, H, W).contiguous()
        # !!! will have to reshape attn_weights into a useable format
        if self.need_weights: return out, attn_weights
        else: return out
    
        
class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, n_hidden):
        super().__init__()
        self.fc1 = nn.Conv2d(n_hidden, n_hidden*4, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(n_hidden*4, n_hidden, kernel_size=1, bias=True)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class ALBERT(nn.Module):
    """ Transformer with Self-Attentive Blocks
    
    choices:
    * type of norm
    * Amount of downsampling before repeated part
    * Position embedding (images need to be MUCH more locally focused than language)    
    * n_layers(repeats)
    * n_hidden
    * activation
    * Size & number of layers before and after repeated part
    
    Assumptions:
    * Swap linear layers for 1x1 conv2d to simplify
    * Bias = true in pointwise since not only using batchnorm
    """
    def __init__(self, n_layers, n_hidden=256, norm='layer', spatial='conv', cifar=True, n_classes=10):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.cifar = cifar
        self.n_classes = n_classes
        self.spatial = spatial
#         self.n_heads=self.n_hidden//32
        
#         if pos_embed_type == 'absolute':
#             self.embedding = 
#         if pos_embed_type == 'absolute_learned':
#             self.embedding = 
#         elif pos_embed_type == 'relative':
#             self.embedding = 

        if norm=='batch':
            self.norm1 = torch.nn.BatchNorm2d(num_features=self.n_hidden)
            self.norm2 = torch.nn.BatchNorm2d(num_features=self.n_hidden)
        elif norm=='layer_albert':
            self.norm1 = LayerNormCustom(self.n_hidden)
            self.norm2 = LayerNormCustom(self.n_hidden)
        elif norm=='layer':
            self.norm1 = torch.nn.LayerNorm([self.n_hidden, 16, 16], eps=1e-05, elementwise_affine=True)
            self.norm2 = torch.nn.LayerNorm([self.n_hidden, 16, 16], eps=1e-05, elementwise_affine=True)
        elif norm=='layer_noaffine':
            self.norm1 = torch.nn.LayerNorm([self.n_hidden, 16, 16], eps=1e-05, elementwise_affine=False)
            self.norm2 = torch.nn.LayerNorm([self.n_hidden, 16, 16], eps=1e-05, elementwise_affine=False)
        elif norm=='group':
            self.norm1 = torch.nn.GroupNorm(num_groups=32, num_channels=self.n_hidden) # standard from their paper
            self.norm2 = torch.nn.GroupNorm(num_groups=32, num_channels=self.n_hidden)
        elif norm=='instance':
            self.norm1 = torch.nn.InstanceNorm2d(num_features=self.n_hidden)
            self.norm2 = torch.nn.InstanceNorm2d(num_features=self.n_hidden)
        elif norm=='weight':
            self.norm1 = torch.nn.utils.weight_norm()
            self.norm2 = torch.nn.utils.weight_norm()
                
                
        if self.cifar:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, self.n_hidden, kernel_size=5, stride=2, padding=2, bias=False)
            self.bn2 = nn.BatchNorm2d(self.n_hidden)
            
        
        self.embed = PositionEmbedding(self.n_hidden, 16, 16)

            
        if self.spatial == 'conv':
            self.spatial_layer = nn.Conv2d(self.n_hidden, self.n_hidden, kernel_size=7, stride=1, padding=3, bias=True)
        if self.spatial == 'attn':
            self.spatial_layer = MultiheadSelfAttention2D(self.n_hidden, num_heads=4, dropout=0.1, bias=True, need_weights=False)
#         elif self.attention_type == 'attention_conv':
#             self.attn = AttentionConv(self.n_hidden, self.n_hidden, kernel_size=5, stride=1, padding=0, groups=1, bias=False)

        self.proj = nn.Conv2d(self.n_hidden, self.n_hidden, kernel_size=1, bias=True)
        self.pwff = PositionWiseFeedForward(self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, self.n_classes)
        self.pool = nn.AdaptiveAvgPool2d(1)
            
    def forward(self, x):
        x = gelu(self.bn1(self.conv1(x)))
        x = gelu(self.bn2(self.conv2(x)))
        # weird spot for position embedding, but I image better to put on downsampled one
        x = self.embed(x) 
        for _ in range(self.n_layers):
#             x = self.attn(x, mask)
            x = self.spatial_layer(x)
            x = self.norm1(x + self.proj(x))
            x = self.norm2(x + self.pwff(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x





