import gudhi
import torch
import os
import numpy as np
import torch.nn as nn

def get_losses(args):
    if args.loss == 'only_topology':
        loss = TopologyLoss(args)
    else:
        raise NotImplementedError

    return loss

class TopologyLoss(nn.Module):
    def __init__(self,args):
        super(TopologyLoss, self).__init__()
        self.args = args


