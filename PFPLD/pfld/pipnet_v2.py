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
import math
import torch.nn.init as init
import torch.nn.functional as F

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

'''
class PFLDInference(nn.Module):
    def __init__(self):
        super(PFLDInference, self).__init__()
        # https://github.com/hanson-young/nniefacelib/issues/27,https://github.com/polarisZhao/PFLD-pytorch/issues/23  pfldx0.25
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv3_1 = InvertedResidual(64, 64, 2, False, 2)

        self.block3_2 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_3 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_4 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_5 = InvertedResidual(64, 64, 1, True, 2)

        self.conv4_1 = InvertedResidual(64, 128, 2, False, 2)

        self.conv5_1 = InvertedResidual(128, 128, 1, False, 4)
        self.block5_2 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_3 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_4 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_5 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_6 = InvertedResidual(128, 128, 1, True, 4)

        self.conv6_1 = InvertedResidual(128, 16, 1, False, 2)  # [16, 14, 14]

        self.conv7 = conv_bn(16, 32, 3, 2)  # [32, 7, 7]
        self.conv8 = nn.Conv2d(32, 128, 7, 1, 0)  # [128, 1, 1]
        self.bn8 = nn.BatchNorm2d(128)

        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)
        self.fc = nn.Linear(176, 196)
        self.fc_aux = nn.Linear(176, 3)

        self.conv1_aux = conv_bn(64, 128, 3, 2)
        self.conv2_aux = conv_bn(128, 128, 3, 1)
        self.conv3_aux = conv_bn(128, 32, 3, 2)
        self.conv4_aux = conv_bn(32, 128, 7, 1)
        self.max_pool1_aux = nn.MaxPool2d(3)
        self.fc1_aux = nn.Linear(128, 32)
        self.fc2_aux = nn.Linear(32 + 176, 3)

    def forward(self, x):  # x: 3, 112, 112
        x = self.relu(self.bn1(self.conv1(x)))  # [64, 56, 56]
        x = self.relu(self.bn2(self.conv2(x)))  # [64, 56, 56]
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        out1 = self.block3_5(x)

        x = self.conv4_1(out1)
        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x = self.block5_5(x)
        x = self.block5_6(x)   ##14  上面是backbone
        x = self.conv6_1(x)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv7(x)  ##7
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.relu(self.conv8(x)) #1
        x3 = x3.view(x1.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)  #176->198   176=16+32+128


        aux = self.conv1_aux(out1)
        aux = self.conv2_aux(aux)
        aux = self.conv3_aux(aux)
        aux = self.conv4_aux(aux)
        aux = self.max_pool1_aux(aux)
        aux = aux.view(aux.size(0), -1)
        aux = self.fc1_aux(aux)
        aux = torch.cat([aux, multi_scale], 1)
        pose = self.fc2_aux(aux)

        return pose, landmarks
'''

class PFLDInference(nn.Module):
    def __init__(self, width_mult=1.0):
        print("Width_mult:", width_mult)
        super(PFLDInference, self).__init__()
        assert width_mult in [0.25, 1.0]
        layer_channels = int(64 * width_mult)
        layer_channels2 = layer_channels * 2
        self.conv1 = nn.Conv2d(
            3, layer_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(layer_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            layer_channels, layer_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(layer_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv3_1 = InvertedResidual(layer_channels, layer_channels, 2, False, 2)

        self.block3_2 = InvertedResidual(layer_channels, layer_channels, 1, True, 2)
        self.block3_3 = InvertedResidual(layer_channels, layer_channels, 1, True, 2)
        self.block3_4 = InvertedResidual(layer_channels, layer_channels, 1, True, 2)
        self.block3_5 = InvertedResidual(layer_channels, layer_channels, 1, True, 2)

        self.conv4_1 = InvertedResidual(layer_channels, layer_channels2, 2, False, 2)

        self.conv5_1 = InvertedResidual(layer_channels2, layer_channels2, 1, False, 4)
        self.block5_2 = InvertedResidual(layer_channels2, layer_channels2, 1, True, 4)
        self.block5_3 = InvertedResidual(layer_channels2, layer_channels2, 1, True, 4)
        self.block5_4 = InvertedResidual(layer_channels2, layer_channels2, 1, True, 4)
        self.block5_5 = InvertedResidual(layer_channels2, layer_channels2, 1, True, 4)
        self.block5_6 = InvertedResidual(layer_channels2, layer_channels2, 1, True, 4)
        
        # PiPNet
        # self.pip_conv = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0)
        self.pip_conv = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.cls_layer = nn.Conv2d(32, 98, kernel_size=1, stride=1, padding=0)
        self.x_layer = nn.Conv2d(32, 98, kernel_size=1, stride=1, padding=0)
        self.y_layer = nn.Conv2d(32, 98, kernel_size=1, stride=1, padding=0)
        self.nb_x_layer = nn.Conv2d(32, 10 * 98, kernel_size=1, stride=1, padding=0)
        self.nb_y_layer = nn.Conv2d(32, 10 * 98, kernel_size=1, stride=1, padding=0)
        # self.pipnet_fc = nn.Flatten()
        # self.linear1 = nn.Linear(110446 , 196)
        # self.linear2 = nn.Linear(196, 196)
        self.relu = nn.ReLU()
        
        self.conv6_1 = InvertedResidual(layer_channels2, 16, 1, False, 2)  # [16, 14, 14]

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
        x = self.relu(self.bn1(self.conv1(x)))  # [64, 56, 56]
        x = self.relu(self.bn2(self.conv2(x)))  # [64, 56, 56]

        x = self.conv3_1(x) # [1, 64, 28, 28]
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        out1 = self.block3_5(x)

        x = self.conv4_1(out1)  # [1, 128, 14, 14]
        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x = self.block5_5(x)
        x = self.block5_6(x)   ##14  上面是backbone
        
        # PiPNet
        # x_pip = self.pip_conv(x)
        x = self.conv6_1(x)  # [1, 16, 14, 14]
        x1 = self.avg_pool1(x)  # [1, 16, 1, 1]
        x1 = x1.view(x1.size(0), -1)    # [1, 16]

        x = self.conv7(x)   # [1, 32, 7, 7]
        
        x_pip = self.pip_conv(x)
        x_1 = self.cls_layer(x_pip)
        x_2 = self.x_layer(x_pip)
        x_3 = self.y_layer(x_pip)
        x_4 = self.nb_x_layer(x_pip)
        x_5 = self.nb_y_layer(x_pip)
        # PiPNet concate
        # x_concate = torch.cat((x_1, x_2, x_3, x_4, x_5), dim=1)
        # x_concate = self.pipnet_fc(x_concate)
        # x_landmark = self.linear1(x_concate)
        # x_landmark = self.linear2(self.relu(x_landmark))
        
        x2 = self.avg_pool2(x)  # [1, 32, 1, 1]
        x2 = x2.view(x2.size(0), -1)
        
        x3 = self.relu(self.conv8(x)) # [1, 128, 1, 1]
        x3 = x3.view(x1.size(0), -1)
        
        multi_scale = torch.cat([x1, x2, x3], 1)
        x_landmark = self.fc(multi_scale)

        aux = self.conv1_aux(out1)
        aux = self.conv2_aux(aux)
        aux = self.conv3_aux(aux)
        aux = self.conv4_aux(aux)
        aux = self.max_pool1_aux(aux)
        aux = aux.view(aux.size(0), -1)
        aux = self.fc1_aux(aux)
        aux = torch.cat([aux, multi_scale], 1)
        pose = self.fc2_aux(aux)

        return x_1, x_2, x_3, x_4, x_5, x_landmark, pose

# class PFLDInference(nn.Module):
#     def __init__(self, resnet, width_mult=1.0):
#         super(PFLDInference, self).__init__()
#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.maxpool = resnet.maxpool
#         self.sigmoid = nn.Sigmoid()
#         self.layer1 = resnet.layer1
#         self.layer2 = resnet.layer2
#         self.layer3 = resnet.layer3
#         self.layer4 = resnet.layer4
        
#         self.cls_layer = nn.Conv2d(256, 98, kernel_size=1, stride=1, padding=0)
#         self.x_layer = nn.Conv2d(256, 98, kernel_size=1, stride=1, padding=0)
#         self.y_layer = nn.Conv2d(256, 98, kernel_size=1, stride=1, padding=0)
#         self.nb_x_layer = nn.Conv2d(256, 10 * 98, kernel_size=1, stride=1, padding=0)
#         self.nb_y_layer = nn.Conv2d(256, 10 * 98, kernel_size=1, stride=1, padding=0)
        
#         nn.init.normal_(self.cls_layer.weight, std=0.001)
#         if self.cls_layer.bias is not None:
#             nn.init.constant_(self.cls_layer.bias, 0)

#         nn.init.normal_(self.x_layer.weight, std=0.001)
#         if self.x_layer.bias is not None:
#             nn.init.constant_(self.x_layer.bias, 0)

#         nn.init.normal_(self.y_layer.weight, std=0.001)
#         if self.y_layer.bias is not None:
#             nn.init.constant_(self.y_layer.bias, 0)

#         nn.init.normal_(self.nb_x_layer.weight, std=0.001)
#         if self.nb_x_layer.bias is not None:
#             nn.init.constant_(self.nb_x_layer.bias, 0)

#         nn.init.normal_(self.nb_y_layer.weight, std=0.001)
#         if self.nb_y_layer.bias is not None:
#             nn.init.constant_(self.nb_y_layer.bias, 0)
        
#     def forward(self, x):  # x: 3, 112, 112
#         # import pdb
#         # pdb.set_trace()
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.maxpool(x) # [128, 64, 28, 28]
#         x = self.layer1(x)
#         x = self.layer2(x)  # [128, 128, 14, 14]
#         x = self.layer3(x)  # [128, 256, 7, 7]
#         x1 = self.cls_layer(x)
#         x2 = self.x_layer(x)
#         x3 = self.y_layer(x)
#         x4 = self.nb_x_layer(x)
#         x5 = self.nb_y_layer(x)
#         return x1, x2, x3, x4, x5, 0

if __name__ == '__main__':
    input = torch.randn(1, 3, 112, 112)
    plfd_backbone = PFLDInference()
    angle, landmarks = plfd_backbone(input)
    print(plfd_backbone)
    print("angle.shape:{0:}, landmarks.shape: {1:}".format(
        angle.shape, landmarks.shape))