import gudhi
import torch
import os
import numpy as np
import torch.nn as nn
from scipy.linalg import circulant
from scipy.spatial.distance import pdist
from manifolds import pmath

def get_losses(args):
    if args.loss == 'only_topology':
        loss = TopologyLoss(args)
    elif args.loss == 'ae':
        loss = AELoss(args)
    elif args.loss == 'regression_poincare':
        loss = PoincareRegressionLoss(args)
    elif args.loss == 'regression_euclidean':
        loss = AELoss(args)
    else:
        raise NotImplementedError

    return loss

class PoincareRegressionLoss(nn.Module):
    def __init__(self, args):
        super(PoincareRegressionLoss, self).__init__()
        self.args = args

    def forward(self, pred, target):
        dis = pmath.dist(pred,target)
        dis = dis.mean()
        return dis


class TopologyLoss(nn.Module):
    def __init__(self,args):
        super(TopologyLoss, self).__init__()
        self.args = args

    def get_hyperbolic_distance_matrix(self, m1):
        m1_extended = circulant(m1[::1,:])

    def forward(self,pred,target):
        pred_distancs = self.get_hyperbolic_distance_matrix(pred)


class AELoss(nn.Module):
    def __init__(self, args):
        super(AELoss, self).__init__()
        self.args = args
        self.mse = nn.MSELoss()
        # self.mse = nn.L1Loss()

    def forward(self, pred, target):
        out = self.mse(pred,target)
        return out