import torch
import numpy as np
import os
import torch.nn.functional as F
import torch.nn as nn
import manifolds
from geoopt.manifolds.stereographic.manifold import PoincareBall
from resae import ResNetAEEncoder, ResNetAEDecoder
import manifolds.pmath as pmath

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
def extra_hidden_layer(hidden_dim, out_dim, non_lin):
    return nn.Sequential(nn.Linear(hidden_dim, out_dim), non_lin)
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
    def __init__(self, manifold, dim, n_f, depth, args):
        super(VideoModel,self).__init__()
        self.manifold = manifold

        self.args = args
        self.dim = dim
        self.n_f = n_f
        self.depth = depth
        self.image_size = 8
        self.window = args.window
        self.enc, self.fc = self.get_layers()



    def get_layers(self):
        layers = []
        non_lin = nn.ReLU(inplace=True)
        layers.append(extra_hidden_layer(self.args.num_extracted_features, self.n_f*self.depth, non_lin))
        for i in range(self.depth,1,-1):
            layers.append(extra_hidden_layer(self.n_f*i,self.n_f*(i-1), non_lin))
            # enc = ResNetAEEncoder(n_levels=self.args.depth, n_ResidualBlock=1, z_dim=self.dim, output_channels=1,
            #                            n_f=self.n_f)
        enc = nn.Sequential(*layers)
        #
        layers_fc = []
        # layers_fc.append(nn.Linear((self.image_size**2)* self.window * self.dim, self.dim))
        layers_fc.append(nn.Linear(self.n_f, self.dim))
        if isinstance(self.manifold, manifolds.PoincareManifold):
            layers_fc.append(ExpMap())
        layers_fc = nn.Sequential(*layers_fc)
        return enc,layers_fc

    def forward(self,x):
        enc_x = self.enc(x)
        enc_x = self.fc(enc_x)
        # enc_x = self.manifold.expmap0(enc_x)

        return enc_x

class ExpMap(nn.Module):
    def __init__(self):
        super(ExpMap, self).__init__()

    def forward(self,x):
        x = pmath.expmap0(x)
        return x

class LogMap(nn.Module):
    def __init__(self):
        super(LogMap, self).__init__()

    def forward(self,x):
        x = pmath.logmap0(x)
        return x

class OxfordModel(nn.Module):
    def __init__(self, manifold, dim, nf, z_dim, args):
        super(OxfordModel, self).__init__()
        self.manifold = manifold

        self.args = args
        self.dim = dim
        self.nf = nf
        self.z_dim = z_dim
        self.mul = 8

        # if self.manifold == 'euclidean':
        #     self.fc = nn.Linear(self.z_dim * self.mul*self.mul, self.dim)
        #     self.fc_dec = nn.Linear( self.dim, self.z_dim *self.mul*self.mul)
        # else:
        #     # self.fc = nn.Sequential(ExpMap())
        #     self.fc = nn.Sequential(nn.Linear(self.z_dim* self.mul*self.mul, self.dim),ExpMap())
        #     self.fc_dec = nn.Sequential(LogMap(),nn.Linear(self.dim, self.z_dim* self.mul*self.mul))
        #     # self.fc_dec = nn.Sequential(LogMap())

        self.enc =  ResNetAEEncoder(n_levels=self.args.depth, n_ResidualBlock=1, z_dim=self.z_dim, output_channels=1, n_f=self.nf)
        self.dec = ResNetAEDecoder(n_levels=self.args.depth, n_ResidualBlock=1, z_dim=self.z_dim, output_channels=1, n_f=self.nf)

    def forward(self,x):
        self.z = self.enc(x)
        self.z = pmath.expmap0(self.z.view(self.args.batch_size,-1))
        # self.z = self.fc(self.z.view(-1,self.z_dim*self.mul*self.mul))
        #
        z_d = pmath.logmap0(self.z).view(-1,self.z_dim,self.mul,self.mul)
        # self.z_d = self.fc_dec(self.z).view(-1,self.z_dim,self.mul,self.mul)
        out = self.dec(z_d)
        return out
