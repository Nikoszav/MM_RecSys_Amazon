
import torch
import torch.nn as nn
from torchvision import models
from transformers import DeiTConfig, DeiTModel, DeiTFeatureExtractor
import config

class ResNet(nn.Module):
    def __init__(self, h_dim):
        super(ResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))         #to not take the last layer
    
    def forward(self, x):
        x = self.resnet(x).squeeze()
        return x
    

def get_feature_extractor(model="resnet", h_dim=None):
    if model=="resnet":
        return ResNet(h_dim)
    elif model=="deit":
        return ResNet(h_dim)
    else:
        raise Exception("Feature extractor model not recognized:", model)