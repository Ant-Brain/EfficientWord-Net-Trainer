import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet_custom import attentive_resnet50
import math

class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()
    def forward(self,x):
        return x

class ResNet50_Classifier(nn.Module):
    def __init__(self, class_count:int) :
        super(ResNet50_Classifier, self).__init__()
        
        self.inp_batchnorm = nn.BatchNorm2d(1)
        
        self.feature_network = torchvision.models.resnet50()
        self.feature_network.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_network.fc = nn.Linear(2048, class_count)

    def forward(self,x) :
        tmp = self.inp_batchnorm(x)
        #print(self.feature_network.bn1(tmp).shape)
        return self.feature_network(tmp)

class ResNetArc_Classifier(nn.Module):
    def __init__(self, model_type:str="resnet50", class_count:int=1000):
        super(ResNetArc_Classifier, self).__init__()
        resnet_model_fn = getattr(torchvision.models, model_type)
        self.inp_batch_norm = nn.BatchNorm2d(1)
        self.feature_network = resnet_model_fn()
        self.feature_network.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_network.fc = nn.Identity()
    
        self.bn_final = nn.BatchNorm1d(2048)

        self.mapper = nn.Linear(2048, class_count, bias=False)
        
        self.s = 8
        self.m = 0
        self.update_margin(self.m, self.s)


    def update_margin(self, m = 0 , s=8):
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

    def forward(self, x ,y=None):
        assert self.s!=0

        x_norm = self.inp_batch_norm(x)
        feat = self.feature_network(x_norm)
        feat_norm = self.bn_final(feat)
        feat_l2 = F.normalize(feat_norm, dim=1)
        
        cosine_dist = torch.matmul(
            feat_l2,
            F.normalize(self.mapper.weight.T, dim=0)
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

        return 8 * out


class AttentiveResNet50Arc_Classifier(nn.Module):
    def __init__(self, class_count:int):
        super(AttentiveResNet50Arc_Classifier, self).__init__()

        self.feature_network = attentive_resnet50()
        #self.feature_network.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_network.fc = EmptyModule()
    
        self.bn_final = nn.BatchNorm1d(2048)

        self.mapper = nn.Linear(2048, class_count, bias=False)

    def forward(self, x):

        feat = self.feature_network(x)
        out = self.bn_final(feat)
        out = 8 * F.normalize(out, dim=1)

        output = torch.matmul(
            out, 
            F.normalize(self.mapper.weight.T, dim=0)
        )

        return output

if __name__=="__main__":
    import torch
    import numpy as np
    import python_speech_features
    
    audio = np.random.rand(24000)
    
    mfcc_features = python_speech_features.mfcc(audio,samplerate=16000,winlen=0.025,winstep=0.01,numcep=32,
                     nfilt=32,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
         ceplifter=64,appendEnergy=True)
    mfcc_features = np.expand_dims(mfcc_features, axis=0)
    
    net = AttentiveResNet50Arc_Classifier(1750)
    #net = efficientformerv2_s0(pretrained=False, input_size=(1,149,64), input_chan=1)
    #net = timm.create_model('efficientformer_l1', in_chans=1)
    net.eval()
    print(net.training)
    inp = np.expand_dims(mfcc_features, axis=0)
    print(inp.shape)
    out = net(torch.Tensor(inp))
    print(out.shape, inp.shape, mfcc_features.shape)