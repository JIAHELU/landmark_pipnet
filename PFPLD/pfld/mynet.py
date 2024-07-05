#!/usr/bin/env python3
# -*- coding:utf-8 -*-

######################################################
#
# pfld.py -
# written by  zhaozhichao and Hanson
#
######################################################

import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import math
import torch.nn.init as init
from pfld.shufflenetv2 import ShuffleNetV2
from pfld.mobilenetv1 import MobileNetV1
from pfld.mobilenetv3 import MobileNetV3_Small

def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inp * expand_ratio,
                inp * expand_ratio,
                3,
                stride,
                1,
                groups=inp * expand_ratio,
                bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



class PFLDInference_M(nn.Module):
    def __init__(self, backbone ='shufflenetv2-0.5',pretrained =False):
        super(PFLDInference_M, self).__init__()
        layer_channels = 128
        layer_channels2 =256
        #backbone = None
        self.relu = nn.ReLU(inplace=True)
        #self.conv6_1 = InvertedResidual(layer_channels2, 16, 1, False, 2)  # [16, 14, 14]
        if backbone =='shufflenetv2-0.5':
            #backbone= ShuffleNetV2([4, 8, 4], [24, 64, 128, 256, 1024]) #replace conv6_1
            layer_channels = 96
            layer_channels2 = 192
            backbone = ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024])
            if pretrained:
                checkpoint = torch.load("./models/shufflenetv2_x0.5-f707e7126e.pth", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint.items():
                    name = k  # remove module.
                    new_state_dict[name] = v
                backbone.load_state_dict(new_state_dict, strict=False)
            self.body = _utils.IntermediateLayerGetter(backbone, {'stage3': 0, 'stage4': 1})

        elif backbone =='mobilenetv1-0.25':
            layer_channels = 128
            layer_channels2 = 256
            backbone = MobileNetV1()
            if pretrained:
                checkpoint = torch.load("./models/mobilenetV1X0.25_pretrain.tar")
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
            self.body = _utils.IntermediateLayerGetter(backbone, {'stage2': 0, 'stage3': 1})
        elif backbone == 'mobilenetv3_small':
            layer_channels = 48
            layer_channels2 = 96
            backbone = MobileNetV3_Small()
            if pretrained:
                checkpoint = torch.load("./models/mbv3_small.pth.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                #load params
                backbone.load_state_dict(new_state_dict)
            self.body = _utils.IntermediateLayerGetter(backbone, {'stage2': 0, 'stage3': 1})
        else:
            backbone = ShuffleNetV2([4, 8, 4], [24, 64, 128, 256, 1024])
            self.body = _utils.IntermediateLayerGetter(backbone, {'stage3': 0, 'stage4': 1})
        self.conv6_1 = conv_bn(layer_channels2, 16, 3, 1)
        self.conv7 = conv_bn(16, 32, 3, 2)  # [32, 7, 7]
        self.conv8 = nn.Conv2d(32, layer_channels2, 7, 1, 0)  # [128, 1, 1]
        self.bn8 = nn.BatchNorm2d(layer_channels2)

        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)
        self.fc = nn.Linear(16+32+layer_channels2, 196)
        self.fc_aux = nn.Linear(16+32+layer_channels2, 3)

        self.conv1_aux = conv_bn(layer_channels, layer_channels2, 3, 2)
        self.conv2_aux = conv_bn(layer_channels2, layer_channels2, 3, 1)
        self.conv3_aux = conv_bn(layer_channels2, 32, 3, 2)
        self.conv4_aux = conv_bn(32, layer_channels2, 7, 1)
        self.max_pool1_aux = nn.MaxPool2d(3)
        self.fc1_aux = nn.Linear(layer_channels2, 32)
        self.fc2_aux = nn.Linear(32 + 16+32+layer_channels2, 3)

    def forward(self, x):  # x: 3, 112, 112
        x_ = self.body(x)   ##14  上面是backbone
        x = self.conv6_1(x_[1])
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv7(x)  ##7
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.relu(self.conv8(x)) #1
        x3 = x3.view(x1.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)  #176->198   176=16+32+128


        aux = self.conv1_aux(x_[0])
        aux = self.conv2_aux(aux)
        aux = self.conv3_aux(aux)
        aux = self.conv4_aux(aux)
        aux = self.max_pool1_aux(aux)
        aux = aux.view(aux.size(0), -1)
        aux = self.fc1_aux(aux)
        aux = torch.cat([aux, multi_scale], 1)
        pose = self.fc2_aux(aux)

        return pose, landmarks

if __name__ == '__main__':
    input = torch.randn(1, 3, 112, 112)
    plfd_backbone = PFLDInference_M()
    angle, landmarks = plfd_backbone(input)
    print(plfd_backbone)
    print("angle.shape:{0:}, landmarks.shape: {1:}".format(
        angle.shape, landmarks.shape))
