import torch
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def sigmoid(x): return (1 + (-x).exp()).reciprocal()
def np_sigmoid(x): return 1.0/(1 + np.exp(-x))
def binary_cross_entropy(pred, y): return -(pred.log()*y + (1-y)*(1-pred).log()).mean()

def weight_BCE(pred,y,weight):
    loss = -(0.05*pred.log()*y + 0.0167*(1-y)*(1-pred).log()).mean()
    return loss

def gaussian_shaped_labels(sigma, sz):
    x, y = np.meshgrid(np.arange(0, sz[0]) - np.floor(float(sz[0]) / 2), np.arange(0, sz[1]) - np.floor(float(sz[1]) / 2))
    d = x ** 2 + y ** 2
    g = np.exp(-0.5 / (sigma ** 2) * d)
    #g = np.roll(g, int(-np.floor(float(sz[0]) / 2.) + 1), axis=0)
    #g = np.roll(g, int(-np.floor(float(sz[1]) / 2.) + 1), axis=1)
    return g.astype(np.float32)

label = np.load('/media/xiankai/Data/tracking/SiamFC-PyTorch/Train/label.npy')
fixed_label = label[0,0,:,:]
instance_weight = torch.ones(fixed_label.shape[0], fixed_label.shape[1])
tmp_idx_P = np.where(fixed_label == 1)
sumP = tmp_idx_P[0].size
tmp_idx_N = np.where(fixed_label == 0)
sumN = tmp_idx_N[0].size
instance_weight[tmp_idx_P] = 0.5 * instance_weight[tmp_idx_P] / sumP
instance_weight[tmp_idx_N] = 0.5 * instance_weight[tmp_idx_N] / sumN
batch_size, n_classes = 10, 4
x = torch.randn(batch_size, n_classes)
target = torch.randint(n_classes, size=(batch_size,), dtype=torch.long)
y = torch.zeros(batch_size, n_classes)
y[range(y.shape[0]), target]=1
pred = torch.sigmoid(x)
loss0 = binary_cross_entropy(pred, y)
print('my loss:',loss0)
fixed_label = y
instance_weight = torch.ones(fixed_label.shape[0], fixed_label.shape[1])
tmp_idx_P = np.where(fixed_label == 1)
sumP = tmp_idx_P[0].size
tmp_idx_N = np.where(fixed_label == 0)
sumN = tmp_idx_N[0].size
instance_weight[tmp_idx_P] = 0.5 * instance_weight[tmp_idx_P] / sumP #0.05
instance_weight[tmp_idx_N] = 0.5 * instance_weight[tmp_idx_N] / sumN #0.0167

loss1 = F.binary_cross_entropy_with_logits(x,y,size_average=False)/batch_size/n_classes
print('standart loss:', loss1)

loss3 = F.binary_cross_entropy_with_logits(x,y,weight = instance_weight, size_average=False)/batch_size/n_classes
print('weight standart loss:',loss3)
sigma = 1.6

loss4 = weight_BCE(pred,y,weight = 0)
print('my weight loss:',loss4)
label = np.load('/media/xiankai/Data/tracking/SiamFC-PyTorch/Train/label.npy')
pred = np.load('/media/xiankai/Data/tracking/SiamFC-PyTorch/Train/output.npy')
label1 = label[0,0,:,:]
fixed_label = gaussian_shaped_labels(sigma,[15, 15])
pred1 = pred[0,0,:,:]
plt.figure(0)
plt.imshow(label1)


plt.figure(1)
plt.imshow(fixed_label)

prediction1 = np_sigmoid(pred1)
plt.figure(2)
plt.imshow(prediction1)

diff = -(label1*np.log(prediction1) + (1-label1)*np.log(1-prediction1))
plt.figure(3)
plt.imshow(diff)