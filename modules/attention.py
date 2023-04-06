import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor
import itertools

def same_padding(kernel:int) :
    """
    Only use when stride=1
    """
    return math.floor((kernel-1)//2)

class SelfAttentionConv2d(nn.Module) :

    def __init__(self, in_channels:int, propagate_val = True) :
        
        super(SelfAttentionConv2d, self).__init__()
        self.propagate_val = propagate_val

        self.query_layer = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)   
        self.key_layer = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.value_layer = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(in_channels)

        self.scale = nn.Parameter(torch.tensor([1.]))
    
    def forward(self, x):

        inp = self.bn(x)
        
        query = self.query_layer(inp)
        key = self.key_layer(inp)

        filter = F.softmax(
            torch.matmul(query, key.transpose(2,3)), dim=1
        )
        
        #print(f"filter Shape:")
        output = self.value_layer(torch.matmul(filter, inp))

        return output + inp

class SelfAttentionConv2dLite(nn.Module):
    def __init__(self,in_channels, propagate_val = True):
        '''
        This version of self attention is really light , but input dimensions need to be fixed
        Cannot be dynamic
        '''        
        super(SelfAttentionConv2dLite, self).__init__()

        attention_channels = math.floor(math.sqrt(in_channels))

        self.out_channels = in_channels

        self.query_layer = nn.Conv1d(in_channels, attention_channels, 1)
        self.key_layer = nn.Conv2d(in_channels, attention_channels, 1)

        self.value_layer = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.scale = nn.Parameter(torch.tensor([0.]))
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

        self.propagate_val = propagate_val

    def forward(self,x) :
        x_size = x.shape

        value = self.value_layer(x) if not(self.propagate_val) else x
        value_flat = self.flatten(value)
        #print(f"Value Shape:{value.shape}, Value_flat:{value_flat.shape}")
        x_flat = self.flatten(x)
        query = self.query_layer(x_flat)
        key = self.query_layer(x_flat)
        #print(query.shape, key.shape)
 
        filter = F.softmax(
            self.scale * torch.matmul(query.transpose(1,2), key), dim=1
        )

        output = torch.matmul(value_flat, filter) + x_flat
        return output.view(-1, self.out_channels, x_size[2], x_size[3]) 

class SelfAttentionConv1d(nn.Module):
    def __init__(self, in_channels, attention_channels, value_kernel=4, attention_kernel=3):
        super(SelfAttentionConv1d, self).__init__()

        self.query_layer = nn.Conv1d(
            in_channels, 
            attention_channels, 
            attention_kernel, 
            padding=floor((attention_kernel-1)//2)
        )
        self.key_layer = nn.Conv1d(
            in_channels, 
            attention_channels, 
            attention_kernel, 
            padding=floor((attention_kernel-1)//2)
        )
        self.value_layer = nn.Conv1d(
            in_channels, 
            in_channels, 
            value_kernel, 
            padding=floor((value_kernel-1)//2)
        )
        self.scale = torch.Tensor([0.])    

    def forward(self,x) :
        
        query = self.query_layer(x)
        key   = self.key_layer(x)
        value = self.value_layer(x)
        
        filter = F.softmax(
            self.scale+torch.matmul(query.transpose(1,2), key) , dim=1
        )

        output = torch.matmul(value, filter) + x

        return output

class SelfAttentionConv1dLite(nn.Module):
    def __init__(self, in_channels, value_kernel=4, attention_kernel=3):
        super(SelfAttentionConv1dLite, self).__init__()

        self.query_layer = nn.Conv1d(
            in_channels, 
            in_channels, 
            attention_kernel, 
            padding=floor((attention_kernel-1)//2)
        )
        self.key_layer = nn.Conv1d(
            in_channels, 
            in_channels, 
            attention_kernel, 
            padding=floor((attention_kernel-1)//2)
        )
        self.value_layer = nn.Conv1d(
            in_channels, 
            in_channels, 
            value_kernel, 
            padding=floor((value_kernel-1)//2)
        )
        self.scale = torch.Tensor([0.])    

    def forward(self,x) :
        
        query = self.query_layer(x)
        key   = self.key_layer(x)
        value = self.value_layer(x)
        
        filter = F.softmax(
            self.scale+torch.matmul(query, key.transpose(1,2)) , dim=1
        )

        output = torch.matmul(filter, value) + x

        return output

class Attention2D(torch.nn.Module):
    def __init__(self, dim=384, key_dim=1, num_heads=4,
                 attn_ratio=2,
                 resolution=7,
                 act_layer=nn.ReLU,
                 stride=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads

        if stride is not None:
            self.resolution = math.ceil(resolution / stride)
            self.stride_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim),
                                             nn.BatchNorm2d(dim), )
            self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
        else:
            self.resolution = resolution
            self.stride_conv = None
            self.upsample = None

        self.N = self.resolution ** 2
        self.N2 = self.N
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.q = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.key_dim, 1),
                               nn.BatchNorm2d(self.num_heads * self.key_dim), )
        self.k = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.key_dim, 1),
                               nn.BatchNorm2d(self.num_heads * self.key_dim), )
        self.v = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.d, 1),
                               nn.BatchNorm2d(self.num_heads * self.d),
                               )
        self.v_local = nn.Sequential(nn.Conv2d(self.num_heads * self.d, self.num_heads * self.d,
                                               kernel_size=3, stride=1, padding=1, groups=self.num_heads * self.d),
                                     nn.BatchNorm2d(self.num_heads * self.d), )
        self.talking_head1 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)
        self.talking_head2 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)

        self.proj = nn.Sequential(act_layer(),
                                  nn.Conv2d(self.dh, dim, 1),
                                  nn.BatchNorm2d(dim), )

        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = x.shape
        if self.stride_conv is not None:
            x = self.stride_conv(x)

        mul = H*W

        q = self.q(x).flatten(2).reshape(B, self.num_heads, -1, mul).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape(B, self.num_heads, -1, mul).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, mul).permute(0, 1, 3, 2)
        
        bias = (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        attn = (
                (q @ k) * self.scale
                #+ bias
        )
        # attn = (q @ k) * self.scale
        attn = self.talking_head1(attn)
        attn = attn.softmax(dim=-1)
        attn = self.talking_head2(attn)
        orig = x


        val2 = (attn @ v)

        out = val2.transpose(2,3).reshape(B, self.dh, -1, W) + v_local
        
        if self.upsample is not None:
            out = self.upsample(out)

        out = self.proj(out)
        return out



if __name__ == "__main__" :
    import numpy as np
    q = torch.rand((2,64,147,24))

    attention_conv2d  = SelfAttentionConv2d(64)
    attention_conv2d_2nd = SelfAttentionConv2dLite(in_channels = 64)
    
    with torch.no_grad() :
        result1 = attention_conv2d(q)
        print(result1.shape)
        
        results2 = attention_conv2d_2nd(q)
        print(results2.shape)
        