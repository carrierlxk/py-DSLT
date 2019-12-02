"""
The architecture of SiamFC
Written by Heng Fan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Config import *


class SiamNet(nn.Module):

    def __init__(self):
        super(SiamNet, self).__init__()

        # architecture (AlexNet like)
        self.feat_extraction1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),             # conv1
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, groups=2),  # conv2, group convolution
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1),           # conv3
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.feat_extraction2 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2), # conv4
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, groups=2) 
        ) # conv5
        # adjust layer as in the original SiamFC in matconvnet
        #self.upsample = 
        self.adjust = nn.Conv2d(1, 1, 1, 1)

        # initialize weights
        self._initialize_weight()

        self.config = Config()

    def forward(self, z, x):
        """
        forward pass
        z: examplare, BxCxHxW
        x: search region, BxCxH1xW1
        """
        # get features for z and x
        z_feat1 = self.feat_extraction1(z)
        x_feat1 = self.feat_extraction1(x)
        mask = self.xcorr(z_feat1,x_feat1)
        _,_, h, w = x.size()
        mask = F.upsample(mask, [h,w], mode = 'bilinear') 
        #F.interpolate(mask,size=(h,w),mode = 'bilinear', align_corners=True)
        # correlation of z and z
        x = x*mask
        #z_feat1 = self.feat_extraction1(z)
        x_feat1 = self.feat_extraction1(x)
        z_feat = self.feat_extraction2(z_feat1)
        x_feat = self.feat_extraction2(x_feat1)
        xcorr_out = self.xcorr(z_feat, x_feat)

        score = self.adjust(xcorr_out)

        return score

    def xcorr(self, z, x):
        """
        correlation layer as in the original SiamFC (convolution process in fact)
        """
        batch_size_x, channel_x, w_x, h_x = x.shape
        x = torch.reshape(x, (1, batch_size_x * channel_x, w_x, h_x))

        # group convolution
        out = F.conv2d(x, z, groups = batch_size_x)

        batch_size_out, channel_out, w_out, h_out = out.shape
        xcorr_out = torch.reshape(out, (channel_out, batch_size_out, w_out, h_out))

        return xcorr_out

    def _initialize_weight(self):
        """
        initialize network parameters
        """
        tmp_layer_idx = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                tmp_layer_idx = tmp_layer_idx + 1
                if tmp_layer_idx < 6:
                    # kaiming initialization
                    nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                else:
                    # initialization for adjust layer as in the original paper
                    m.weight.data.fill_(1e-3)
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def weight_loss(self, prediction, label, weight):
        """
        weighted cross entropy loss
        """
        alpha = 0.25
        gamma = 2
        p = prediction.sigmoid()
        pt = p*label + (1-p)*(1-label)         # pt = p if t > 0 else 1-p
        w = alpha*label + (1-alpha)*(1-label)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(prediction,
                                                  label,
                                                  weight,
                                                  size_average=False) / self.config.batch_size
