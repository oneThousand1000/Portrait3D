import torch
import torch.nn as nn
from torch.nn import functional as F

from config import cfg

def make_linear_layers(feat_dims, relu_final=True, use_bn=False):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_conv1d_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv1d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_deconv_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.ConvTranspose2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class GraphConvBlock(nn.Module):
    def __init__(self, adj, dim_in, dim_out):
        super(GraphConvBlock, self).__init__()
        self.adj = adj
        self.vertex_num = adj.shape[0]
        self.fcbn_list = nn.ModuleList([nn.Sequential(*[nn.Linear(dim_in, dim_out), nn.BatchNorm1d(dim_out)]) for _ in range(self.vertex_num)])

    def forward(self, feat):
        batch_size = feat.shape[0]

        # apply kernel for each vertex
        feat = torch.stack([fcbn(feat[:,i,:]) for i,fcbn in enumerate(self.fcbn_list)],1)

        # apply adj
        adj = self.adj.cuda()[None,:,:].repeat(batch_size,1,1)
        feat = torch.bmm(adj, feat)

        # apply activation function
        out = F.relu(feat)
        return out


class GraphResBlock(nn.Module):
    def __init__(self, adj, dim):
        super(GraphResBlock, self).__init__()
        self.adj = adj
        self.graph_block1 = GraphConvBlock(adj, dim, dim)
        self.graph_block2 = GraphConvBlock(adj, dim, dim)

    def forward(self, feat):
        feat_out = self.graph_block1(feat)
        feat_out = self.graph_block2(feat_out)
        out = feat_out + feat
        return out
