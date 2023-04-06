import math
from torchsummary import summary

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.mobilenet_block import h_swish
from modules.mobilenet_block import conv_3x3_bn, conv_2x2_bn, conv_1x1_bn, InvertedResidual2D, SELayer
from modules.attention import SelfAttentionConv2d, SelfAttentionConv2dLite

class EmptyModule(nn.Module) :
    def __init__(self, *args) :
        super(EmptyModule, self).__init__()

    def forward(self, x) :
        return x

class AttentiveMobileWordFeature(nn.Module) :
    def __init__(self, ) :
        super(AttentiveMobileWordFeature, self).__init__()        

        self.layers_cfg = [
            (conv_3x3_bn, (1, 16 ,2)),
            (InvertedResidual2D, (16 , 1.0, 16, 3, 1, EmptyModule , True)), 
            (InvertedResidual2D, (16 , 2.25, 24, 3, 1, EmptyModule , True)),
            (InvertedResidual2D, (24, 1.835, 24, 3, 1, EmptyModule, True)),
            (InvertedResidual2D, (24, 2, 40, 5, 2, SelfAttentionConv2dLite, True)),
            (InvertedResidual2D, (40, 3, 40, 5, 1, EmptyModule, True)), 
            (InvertedResidual2D, (40, 3, 40, 5, 1, EmptyModule, True)), 
            (InvertedResidual2D, (40, 1.5, 48, 5, 1, EmptyModule, True)), 
            (InvertedResidual2D, (48, 1.5, 48, 5, 1, SelfAttentionConv2dLite, True)),
            (InvertedResidual2D, (48, 3, 96, 5, 2, EmptyModule, True)), 
            (InvertedResidual2D, (96, 3, 96, 5, 1, EmptyModule, True)), 
            (InvertedResidual2D, (96, 3, 96, 5, 1, EmptyModule, True)),
            (SelfAttentionConv2dLite , (96,)),
            (conv_1x1_bn, (96, 576)),
            (nn.AvgPool2d, (8,))
            #(SelfAttentionConv2dLite, (96,))

        ]

        self.output_feature_count = 1152

        self.initialize_layers()
    
    def change_layers_cfg(self, layers_cfg) :
        self.layers_cfg = layers_cfg
        self.initialize_layers()

    def initialize_layers(self) :

        self.layers = nn.ModuleList([
            nn.BatchNorm2d(1)
        ])
    
        for layer_cfg in self.layers_cfg :
            #print(layer_cfg[0])
            self.layers.append(
                layer_cfg[0](
                    *layer_cfg[1]
                )
            )


        self.last_layers = nn.ModuleList([
            nn.Flatten(start_dim=1),
            nn.BatchNorm1d(1152),
        ])

    def forward(self, x) :

        interim = x
        for i, layer in enumerate(self.layers) :
            #print(i, layer)
            interim = layer(interim)
            #print(interim.shape)
        for i, layer in enumerate(self.last_layers):
            #print(i, layer)
            #print(interim.shape)
            interim = layer(interim)
            #print(interim.shape)
        out = interim
        return out #out.view(-1,1,out.shape[1])

class AttentiveMobileWordClassifier(nn.Module) :
    def __init__(self, num_classes, m = 0.5, s= 30) :
        super(AttentiveMobileWordClassifier, self).__init__()
        self.feature_network = AttentiveMobileWordFeature()
        self.mapper = nn.Parameter(torch.FloatTensor(self.feature_network.output_feature_count, num_classes))        
        nn.init.xavier_uniform_(self.mapper)
        self.update_margin(m, s)
    
    def update_margin(self, m, s):
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.s = s
        self.m = m
    
    def forward(self, x, y=None) :

        #if y!=None :
            #self.mapper = nn.Parameter(F.normalize(self.mapper.data, dim=0))
            #self.mapper = F.normalize(self.mapper, dim=0)

        feature = self.feature_network(x)
        
        cosine_dist =  torch.matmul(
            F.normalize(feature, dim=1), 
            self.mapper
        )

        sine_dist = torch.sqrt(1 - torch.square(cosine_dist))#.clamp(-1,1)

        out_dist = cosine_dist

        if y!=None :

            margined_cosine_dist = cosine_dist*self.cos_m - sine_dist*self.sin_m # cos(a+b) eqn
        
            margined_cosine_dist = torch.where(
                cosine_dist>0, 
                margined_cosine_dist, 
                cosine_dist
            ) # easy margin

            out_dist = margined_cosine_dist*y + cosine_dist*(1-y)
        
        out = self.s * out_dist
        
        return out


class AttentiveMobileWordClassifier_Linear(nn.Module) :
    def __init__(self, num_classes, m = 0.5, s= 30) :
        super(AttentiveMobileWordClassifier_Linear, self).__init__()
        self.feature_network = AttentiveMobileWordFeature()
        self.mapper = nn.Linear(self.feature_network.output_feature_count, num_classes)
        #self.mapper = nn.Parameter(torch.FloatTensor(self.feature_network.output_feature_count, num_classes))        
        #nn.init.xavier_uniform_(self.mapper)
        #self.update_margin(m, s)
    
    def update_margin(self, m, s):
        pass
    
    def forward(self, x, y=None) :

        #if y!=None :
        #    self.mapper = F.normalize(self.mapper, dim=0)

        feature = self.feature_network(x)
        feature = F.normalize(feature, dim=1) 
        out = self.mapper(feature)
        return out

#class AttentiveMobileWordClassifierV2(nn.Module):
#c    def __init__

import numpy as np
import python_speech_features

audio = np.random.rand(24000)

mfcc_features = python_speech_features.mfcc(audio,samplerate=16000,winlen=0.025,winstep=0.01,numcep=64,
                 nfilt=64,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
     ceplifter=64,appendEnergy=True)
mfcc_features = np.expand_dims(mfcc_features, axis=0)

from efficientformer_v2 import efficientformerv2_s0
import timm
net = AttentiveMobileWordClassifier(2354)
#net = efficientformerv2_s0(pretrained=False, input_size=(1,149,64), input_chan=1)
#net = timm.create_model('efficientformer_l1', in_chans=1)
net.eval()
#print(net.training)
inp = np.expand_dims(mfcc_features, axis=0)
print(inp.shape)
out = net(torch.Tensor(inp))
print(out.shape, inp.shape, mfcc_features.shape)