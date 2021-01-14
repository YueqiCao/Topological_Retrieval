import torch as th
from torch.autograd import Function
from torch.utils.data import DataLoader


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


def get_dataloader(dataset, shuffle=False, workers=0):

    kwargs = {'num_workers': workers, 'pin_memory': True, 'shuffle': shuffle}
    loader = DataLoader(dataset, batch_size=1, **kwargs)

    return loader