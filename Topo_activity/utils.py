import torch as th
from torch.autograd import Function
from torch.utils.data import DataLoader
import pickle

class Acosh(Function):
    @staticmethod
    def forward(ctx, x, eps):
        z = th.sqrt(x * x - 1)
        ctx.save_for_backward(z)
        ctx.eps = eps
        return th.log(x + z)

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        z = th.clamp(z, min=ctx.eps)
        z = g / z
        return z, None


acosh = Acosh.apply


def get_dataloader(dataset, shuffle=False, workers=0, batch_size=32):

    kwargs = {'num_workers': workers, 'pin_memory': True, 'shuffle': shuffle}
    loader = DataLoader(dataset, batch_size=batch_size, **kwargs)

    return loader


def pickle_object(object, outpath):
    fp = open(outpath, "wb")
    pickle.dump(object, fp, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle_object(path):
    with open(path, 'rb') as handle:
        b = pickle.load(handle)
    return b
