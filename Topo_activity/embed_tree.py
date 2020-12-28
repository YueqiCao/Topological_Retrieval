import torch
import numpy as np
import os
from data.dataloader import TreeDataset
from model import get_model
import argparse
import logging
from rsgd import RiemannianSGD

def run_train(args):
    dataset = TreeDataset(args.json_path, negs=args.negs,
                          batch_size=args.batch_size,
                          ndproc=args.ndproc,
                          burnin=args.burnin,
                          dampening=args.dampening,
                          neg_multiplier=args.neg_multiplier)

    data = dataset.get_graph_dataset()

    model = get_model(args,len(dataset.objects))
    optimizer = RiemannianSGD(model.optim_params(), lr=args.lr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    # Logging
    parser.add_argument('--exp_name', type=str, default='dev')
    parser.add_argument('--exp_root', type=str, default='./experiments')
    parser.add_argument('--restore', type=bool, default=False)
    # Dataset
    parser.add_argument('--json_path', type=str, default="../hyperbolic_action-master/activity_net.v1-3.json")
    parser.add_argument('--negs', type=int, default=50,help='negative samples')
    parser.add_argument('--ndproc', type=int, default=4,help='Number of data loading processes')
    parser.add_argument('--dampening', type=float, default=0.75,help='Sample dampening during burnin')
    parser.add_argument('--neg_multiplier', type=float, default=1.0)

    # Model
    parser.add_argument('-dim', type=int, default=5, help='Embedding dimension')
    parser.add_argument('-manifold', type=str, default='lorentz')
    parser.add_argument('-model', type=str, default='distance', help='Energy function model')
    parser.add_argument('-margin', type=float, default=0.1, help='Hinge margin')

    # Optimization
    parser.add_argument('-sparse', default=False, action='store_true', help='Use sparse gradients for embedding table')
    parser.add_argument('-lr', type=float, default=0.3, help='Learning rate')
    # General
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--burnin', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=100)
    # OS
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.seed == -1: args.seed = int(torch.randint(0, 2 ** 32 - 1, (1,)).item())
    print('seed', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.restore:
        oldrunId = args.exp_name
        args.exp_name = args.exp_name + '_cont'

    args.logdir = os.path.join(args.exp_root, args.exp_name)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    run_train(args)