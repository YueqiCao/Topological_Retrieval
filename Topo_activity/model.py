import torch
import numpy as np
import os
import torch.nn.functional as F
import torch.nn as nn

import manifolds
from geoopt.manifolds.stereographic.manifold import PoincareBall

MANIFOLDS = {
    'lorentz': manifolds.LorentzManifold,
    'poincare': manifolds.PoincareManifold,
    'euclidean': manifolds.EuclideanManifold,
    'geoopt_poincare': PoincareBall
}
MODELS = {
    'distance': 'DistanceEnergyFunction',
    'entailment_cones': 'EntailmentConeEnergyFunction',
}
def extra_hidden_layer_conv(in_dim, out_dim, non_lin, stride):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, 3, stride, 1), non_lin)

def get_model(args, N):
    K = 0.1 if args.model == 'entailment_cones' else None
    manifold = MANIFOLDS[args.manifold](K=K)
    return eval(MODELS[args.model])(
        manifold,
        dim=args.dim,
        size=N,
        sparse=args.sparse,
        margin=args.margin
    )


class EnergyFunction(torch.nn.Module):
    def __init__(self, manifold, dim, size, sparse=False, **kwargs):
        super().__init__()
        self.manifold = manifold
        self.lt = manifold.allocate_lt(size, dim, sparse)
        self.nobjects = size
        self.manifold.init_weights(self.lt)

    def forward(self, inputs):
        e = self.lt(inputs)
        with torch.no_grad():
            e = self.manifold.normalize(e)
        o = e.narrow(1, 1, e.size(1) - 1)
        s = e.narrow(1, 0, 1).expand_as(o)
        return self.energy(s, o).squeeze(-1)

    def optim_params(self):
        return [{
            'params': self.lt.parameters(),
            'rgrad': self.manifold.rgrad,
            'expm': self.manifold.expm,
            'logm': self.manifold.logm,
            'ptransp': self.manifold.ptransp,
        }]

    def loss_function(self, inp, target, **kwargs):
        raise NotImplementedError


class DistanceEnergyFunction(EnergyFunction):
    def energy(self, s, o):
        return self.manifold.distance(s, o)

    def loss(self, inp, target, **kwargs):
        return F.cross_entropy(inp.neg(), target)


class EntailmentConeEnergyFunction(EnergyFunction):
    def __init__(self, *args, margin=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.manifold.K is not None, (
            "K cannot be none for EntailmentConeEnergyFunction"
        )
        assert hasattr(self.manifold, 'angle_at_u'), 'Missing `angle_at_u` method'
        self.margin = margin

    def energy(self, s, o):
        energy = self.manifold.angle_at_u(o, s) - self.manifold.half_aperture(o)
        return energy.clamp(min=0)

    def loss(self, inp, target, **kwargs):
        loss = inp[:, 0].clamp_(min=0).sum()  # positive
        loss += (self.margin - inp[:, 1:]).clamp_(min=0).sum()  # negative
        return loss / inp.numel()

class VideoModel(nn.Module):
    def __init__(self, manifold, dim, size, depth, args):
        super(VideoModel,self).__init__()
        self.manifold = MANIFOLDS[manifold]

        self.args = args
        self.dim = dim
        self.size = size
        self.depth = depth

        self.enc = self.get_layers()
        self.fc = nn.Linear(self.size*self.depth,self.dim)

    def get_layers(self):
        layers = []
        non_lin = nn.ReLU(inplace=True)

        for i in range(1,self.depth):
            layers.append(extra_hidden_layer_conv(self.size*i,self.size*(i+1), non_lin, 2))

        layers = nn.Sequential(*layers)
        return layers

    def forward(self,x):
        enc_x = self.enc(x)
        enc_x = self.fc(enc_x)
        riem_x = self.manifold.expmap0(enc_x)

        return riem_x
