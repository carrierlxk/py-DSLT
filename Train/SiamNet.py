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
        self.feat_extraction = nn.Sequential(
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
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, groups=2), # conv4
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, groups=2)  # conv5
        )

        # adjust layer as in the original SiamFC in matconvnet
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
        z_feat = self.feat_extraction(z)
        x_feat = self.feat_extraction(x)

        # correlation of z and z
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
      
    def sigmoid(self,x): return (1 + (-x).exp()).reciprocal()
         
    def weight_BCE(self,pred,y,weight):
        loss = -(torch.max(weight)*pred.log()*y + torch.min(weight)*(1-y)*(1-pred).log()).mean()
        return loss 
    
    def shrinkage(self, pred):
        k = 10#10#6#10
        a = 0.6#0.7#0.6#0.5#
        pred1 = (1+((k*(pred-a))).exp()).reciprocal() * pred.exp()
        #pred1 = (1+((k*(pred-a))).exp()).reciprocal() 
        return pred1
    
    #def shrinkage1(self, pred):
        

    def weight_loss(self, prediction, label, weight):
        """
        weighted cross entropy loss
        """
        #loss =  F.binary_cross_entropy_with_logits(prediction,
        #                                          label.clone(),
        #                                          weight,
        #                                          size_average=False) / self.config.batch_size
        prediction = prediction.clamp(min=-10, max=10)
        #print('range:',torch.max(prediction), torch.min(prediction))
        prediction1 = self.sigmoid(prediction).clone()
        criterion = nn.MSELoss(reduce=False, size_average = True).cuda()
        #alpha = 0.75
        gamma = 2
        #pt = prediction1*(label) + (1-prediction1)*(1-label)        # pt = p if t > 0 else 1-p
        #w = alpha*label + (1-alpha)*(1-label)  # w = alpha if t > 0 else 1-alpha
        #w = w * (1-pt).pow(gamma)#
        #prediction1 = torch.clamp(prediction1,min=1e-8,max=1-1e-8)
        #print('value:', torch.min(w), torch.max(w))
        #loss1 = (w*(-(prediction1.log()*label + (1-label)*(1-prediction1).log()))).mean()
        loss2 = -(torch.max(weight)*(1-prediction1)**(gamma)*prediction1.log()*label + torch.min(weight)*(prediction1)**(gamma)*(1-label)*(1-prediction1).log()).mean()#*255.0
       # loss3 = -(torch.max(weight)*prediction1.log()*label + torch.min(weight)*(1-label)*(1-prediction1).log()).mean()*255.0
        #print('loss:',  (loss1), (loss2))
        #loss4 = -(torch.max(weight)*(self.shrinkage(prediction1))*prediction1.log()*label + torch.min(weight)*(self.shrinkage(1-prediction1))*(1-label)*(1-prediction1).log()).mean()*255.0
        #loss5 = -(torch.max(weight)*(self.shrinkage(prediction1))*prediction1.log()*label + torch.min(weight)*(self.shrinkage(1-prediction1))*(1-label)*(1-prediction1).log()).mean()*255.0
        #loss = criterion(prediction, label)
        #loss = loss*weight
        #print('loss shape:', loss.size())
        #loss1 = torch.sum((prediction1-label)**2) # size_average =False equal to MSELoss
        #print('loss', loss,loss1)
       
        return loss2
