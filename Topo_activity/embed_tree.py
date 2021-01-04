import torch
import numpy as np
import os
from data.dataloader import TreeDataset
from model import get_model
import argparse
import logging
from rsgd import RiemannianSGD
from checkpoint import LocalCheckpoint
import torch.multiprocessing as mp
import sys
import json
import shutil
from train_model import train
from eval_utils import eval_reconstruction
from hypernymy_eval import main as hype_eval

def reconstruction_eval(adj, opt, epoch, elapsed, loss, pth, best):
    chkpnt = torch.load(pth, map_location='cpu')
    model = get_model(opt, chkpnt['embeddings'].size(0))
    model.load_state_dict(chkpnt['model'])

    meanrank, maprank = eval_reconstruction(adj, model)
    sqnorms = model.manifold.norm(model.lt)
    return {
        'epoch': epoch,
        'elapsed': elapsed,
        'loss': loss,
        'sqnorm_min': sqnorms.min().item(),
        'sqnorm_avg': sqnorms.mean().item(),
        'sqnorm_max': sqnorms.max().item(),
        'mean_rank': meanrank,
        'map_rank': maprank,
        'best': bool(best is None or loss < best['loss']),
    }


def hypernymy_eval(epoch, elapsed, loss, pth, best):
    _, summary = hype_eval(pth, cpu=True)
    return {
        'epoch': epoch,
        'elapsed': elapsed,
        'loss': loss,
        'best': bool(
            best is None or summary['eval_hypernymy_avg'] > best['eval_hypernymy_avg'])
        ,
        **summary
    }

def async_eval(adj, q, logQ, opt):
    best = None
    while True:
        temp = q.get()
        if temp is None:
            return

        if not q.empty():
            continue

        epoch, elapsed, loss, pth = temp
        if opt.eval == 'reconstruction':
            lmsg = reconstruction_eval(adj, opt, epoch, elapsed, loss, pth, best)
        elif opt.eval == 'hypernymy':
            lmsg = hypernymy_eval(epoch, elapsed, loss, pth, best)
        else:
            raise ValueError(f'Unrecognized evaluation: {opt.eval}')
        best = lmsg if lmsg['best'] else best
        logQ.put((lmsg, pth))


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
    # setup checkpoint
    checkpoint = LocalCheckpoint(
        args.checkpoint,
        include_in_all={ 'conf' : vars(args).copy(), 'objects' : data.objects},
        start_fresh=True
    )

    state = checkpoint.initialize({'epoch': 0, 'model': model.state_dict()})
    model.load_state_dict(state['model'])
    args.epoch_start = state['epoch']

    adj = {}
    for inputs, _ in data:
        for row in inputs:
            x = row[0].item()
            y = row[1].item()
            if x in adj:
                adj[x].add(y)
            else:
                adj[x] = {y}

    controlQ, logQ = mp.Queue(), mp.Queue()
    control_thread = mp.Process(target=async_eval, args=(adj, controlQ, logQ, args))
    control_thread.start()

    # control closure
    def control(model, epoch, elapsed, loss):
        """
        Control thread to evaluate embedding
        """
        lt = model.w_avg if hasattr(model, 'w_avg') else model.lt.weight.data
        model.manifold.normalize(lt)

        checkpoint.path = f'{args.checkpoint}.{epoch}'
        checkpoint.save({
            'model': model.state_dict(),
            'embeddings': lt,
            'epoch': epoch,
            'model_type': args.model,
        })

        controlQ.put((epoch, elapsed, loss, checkpoint.path))

        while not logQ.empty():
            lmsg, pth = logQ.get()
            shutil.move(pth, args.checkpoint)
            if lmsg['best']:
                shutil.copy(args.checkpoint, args.checkpoint + '.best')
            args.log.info(f'json_stats: {json.dumps(lmsg)}')

    control.checkpoint = True
    model = model.to(args.device)
    if hasattr(model, 'w_avg'):
        model.w_avg = model.w_avg.to(args.device)
    if args.train_threads > 1:
        threads = []
        model = model.share_memory()
        args_ = (args.device, model, data, optimizer, args, log)
        kwargs = {'ctrl': control, 'progress' : not args.quiet}
        for i in range(args.train_threads):
            kwargs['rank'] = i
            threads.append(mp.Process(target=train, args=args_, kwargs=kwargs))
            threads[-1].start()
        [t.join() for t in threads]
    else:
        train(args.device, model, data, optimizer, args, log, ctrl=control,
            progress=not args.quiet)
    controlQ.put(None)
    control_thread.join()
    while not logQ.empty():
        lmsg, pth = logQ.get()
        shutil.move(pth, args.checkpoint)
        log.info(f'json_stats: {json.dumps(lmsg)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    # Logging
    parser.add_argument('--exp_name', type=str, default='activity_net')
    parser.add_argument('--exp_root', type=str, default='./experiments')
    parser.add_argument('--restore', type=bool, default=False)
    # Dataset
    parser.add_argument('--json_path', type=str, default="../hyperbolic_action-master/activity_net.v1-3.json")
    parser.add_argument('--negs', type=int, default=50,help='negative samples')
    parser.add_argument('--ndproc', type=int, default=4,help='Number of data loading processes')
    parser.add_argument('--dampening', type=float, default=0.75,help='Sample dampening during burnin')
    parser.add_argument('--neg_multiplier', type=float, default=1.0)

    # Model
    parser.add_argument('--dim', type=int, default=5, help='Embedding dimension')
    parser.add_argument('--manifold', type=str, default='lorentz')
    parser.add_argument('--model', type=str, default='distance', help='Energy function model')
    parser.add_argument('--margin', type=float, default=0.1, help='Hinge margin')
    parser.add_argument('--eval', choices=['reconstruction', 'hypernymy'],
                        default='reconstruction', help='Which type of eval to perform')

    # Optimization
    parser.add_argument('--sparse', default=True, action='store_true', help='Use sparse gradients for embedding table')
    parser.add_argument('--lr', type=float, default=0.3, help='Learning rate')
    # General
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--quiet', action='store_true', default=False)
    parser.add_argument('--eval_each', type=int, default=1,
                        help='Run evaluation every n-th epoch')
    parser.add_argument('--burnin', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=800)
    # OS
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_threads', type=int, default=1,
                        help='Number of threads to use in training')
    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

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

    args.checkpoint = os.path.join(args.logdir, args.exp_name)

    log_level = logging.INFO
    log = logging.getLogger('lorentz')
    logging.basicConfig(level=log_level, format='%(message)s', stream=sys.stdout)
    args.log = log

    # with open(os.path.join(args.logdir,'config.json'),'w') as f :
    #     t  = vars(args)
    #     del t['device'],
    #     del t['log']
    #     json.dump(t, f)

    run_train(args)