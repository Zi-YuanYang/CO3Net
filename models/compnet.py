import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import math
import warnings


class GaborConv2d(nn.Module):
    '''
    DESCRIPTION: an implementation of the Learnable Gabor Convolution (LGC) layer \n
    INPUTS: \n
    channel_in: should be 1 \n
    channel_out: number of the output channels \n
    kernel_size, stride, padding: 2D convolution parameters \n
    init_ratio: scale factor of the initial parameters (receptive filed) \n
    '''
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, init_ratio=1):
        super(GaborConv2d, self).__init__()

        # assert channel_in == 1

        self.channel_in = channel_in
        self.channel_out = channel_out

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding      

        self.init_ratio = init_ratio 

        self.kernel = 0

        if init_ratio <=0:
            init_ratio = 1.0
            print('input error!!!, require init_ratio > 0.0, using default...')

        # initial parameters
        self._SIGMA = 9.2 * self.init_ratio
        self._FREQ = 0.057 / self.init_ratio
        self._GAMMA = 2.0

        
        # shape & scale of the Gaussian functioin:
        self.gamma = nn.Parameter(torch.FloatTensor([self._GAMMA]), requires_grad=True)          
        self.sigma = nn.Parameter(torch.FloatTensor([self._SIGMA]), requires_grad=True)
        self.theta = nn.Parameter(torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out, requires_grad=False)

        # frequency of the cosine envolope:
        self.f = nn.Parameter(torch.FloatTensor([self._FREQ]), requires_grad=True)
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)


    def genGaborBank(self, kernel_size, channel_in, channel_out, sigma, gamma, theta, f, psi):
        xmax = kernel_size // 2
        ymax = kernel_size // 2
        xmin = -xmax
        ymin = -ymax

        ksize = xmax - xmin + 1
        y_0 = torch.arange(ymin, ymax + 1).float()    
        x_0 = torch.arange(xmin, xmax + 1).float()

        # [channel_out, channelin, kernel_H, kernel_W]   
        y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1) 
        x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize) 

        x = x.float().to(sigma.device)
        y = y.float().to(sigma.device)

        # x=x.float()
        # y=y.float()

        # Rotated coordinate systems
        # [channel_out, <channel_in, kernel, kernel>], broadcasting
        x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
        y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))  
                
        gb = -torch.exp(
            -0.5 * ((gamma * x_theta) ** 2 + y_theta ** 2) / (8*sigma.view(-1, 1, 1, 1) ** 2)) \
            * torch.cos(2 * math.pi * f.view(-1, 1, 1, 1) * x_theta + psi.view(-1, 1, 1, 1))
    
        gb = gb - gb.mean(dim=[2,3], keepdim=True)

        return gb


    def forward(self, x):
        kernel = self.genGaborBank(self.kernel_size, self.channel_in, self.channel_out, self.sigma, self.gamma, self.theta, self.f, self.psi)
        self.kernel = kernel
        # print(x.shape)
        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)

        return out

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=1):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class CompetitiveBlock(nn.Module):
    '''
    DESCRIPTION: an implementation of the Competitive Block::
    
    [CB = LGC + argmax + PPU] \n

    INPUTS: \n

    channel_in: only support 1 \n
    n_competitor: number of channels of the LGC (channel_out)  \n

    ksize, stride, padding: 2D convolution parameters \n

    init_ratio: scale factor of the initial parameters (receptive filed) \n

    o1, o2: numbers of channels of the conv_1 and conv_2 layers in the PPU, respectively. (PPU parameters)
    '''
    def __init__(self, channel_in, n_competitor, ksize, stride, padding, init_ratio=1, o1=32, o2=12):
        super(CompetitiveBlock, self).__init__()

        self.channel_in = channel_in
        self.n_competitor = n_competitor         

        self.init_ratio = init_ratio

        # LGC
        self.gabor_conv2d = GaborConv2d(channel_in = channel_in, channel_out = n_competitor, kernel_size=ksize, stride=stride, padding=ksize//2, init_ratio=init_ratio)
        self.gabor_conv2d2 = GaborConv2d(channel_in = n_competitor, channel_out = n_competitor, kernel_size=ksize, stride=1, padding=ksize//2, init_ratio=init_ratio)

        self.cooratt1 = CoordAtt(n_competitor,n_competitor)
        self.cooratt2 = CoordAtt(n_competitor,n_competitor)

        # soft-argmax
        self.a = nn.Parameter(torch.FloatTensor([1]))
        self.b = nn.Parameter(torch.FloatTensor([0]))
        self.argmax = nn.Softmax(dim=1)

        # PPU
        self.conv1 = nn.Conv2d(n_competitor, o1, 5, 1, 0)
        self.maxpool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(o1, o2, 1, 1, 0)

    def forward(self, x):
        x = self.gabor_conv2d(x)

        x = self.cooratt1(x)
        x = self.gabor_conv2d2(x)
        x = self.cooratt2(x)

        x = (x-self.b)*self.a

        x = self.argmax(x)


        x = self.conv1(x)        
        x = self.maxpool(x)
        x = self.conv2(x)

        return x

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance::
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)

        From: https://github.com/ronghuaiyang/arcface-pytorch
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        if self.training :
            assert label is not None
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

            phi = cosine * self.cos_m - sine * self.sin_m

            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)       
            
            one_hot = torch.zeros(cosine.size(), device=cosine.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)

            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  
            output *= self.s
        else:
            # assert label is None
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            output = self.s * cosine

        return output

class co3net(torch.nn.Module):
    '''
    CompNet = CB1//CB2//CB3 + FC + Dropout + (angular_margin) Output\n
    https://ieeexplore.ieee.org/document/9512475
    '''

    def __init__(self, num_classes):
        super(co3net, self).__init__()

        self.num_classes = num_classes

        self.cb1 = CompetitiveBlock(channel_in=1, n_competitor=9, ksize=35, stride=3, padding=17, init_ratio=1)
        self.cb2 = CompetitiveBlock(channel_in=1, n_competitor=36, ksize=17, stride=3, padding=8, init_ratio=0.5, o2=24)
        self.cb3 = CompetitiveBlock(channel_in=1, n_competitor=9, ksize=7, stride=3, padding=3, init_ratio=0.25)

        self.fc = torch.nn.Linear(17328, 4096)  # <---
        self.fc1 = torch.nn.Linear(4096, 2048)
        self.drop = torch.nn.Dropout(p=0.5)
        # self.arclayer = torch.nn.Linear(1024,num_classes)
        self.arclayer = ArcMarginProduct(2048, num_classes, s=30, m=0.5, easy_margin=False)

    def forward(self, x, y=None):
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)
        x = torch.cat((x1, x2, x3), dim=1)

        # g = self.g1(x)

        x1 = self.fc(x)
        x = self.fc1(x1)
        fe = torch.cat((x1,x),dim=1)
        x = self.drop(x)
        x = self.arclayer(x, y)

        return x, F.normalize(fe, dim=-1)

    def getFeatureCode(self, x):
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)
        x = torch.cat((x1, x2, x3), dim=1)

        x = self.fc(x)
        x = self.fc1(x)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)

        return x
