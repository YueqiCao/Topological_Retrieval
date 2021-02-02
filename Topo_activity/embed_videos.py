import torch
import os
import numpy as np
import pandas as pd
import logging
import argparse
from tqdm import tqdm
from data.dataloader import VideoDataset, FeatureVideoDataset
from model import VideoModel, get_model, MANIFOLDS
import sys
from utils import get_dataloader
from losses import get_losses
from checkpoint import LocalCheckpoint
from torch.utils.tensorboard import SummaryWriter
from geoopt.manifolds.stereographic.manifold import PoincareBall




def run_train(args):
    writer = SummaryWriter(args.logdir,flush_secs=1)

    # Get Dataloaders
    if args.use_extracted_features:
        train_dataset = FeatureVideoDataset(args.feature_path,mode='training',args=args)
        val_dataset = FeatureVideoDataset(args.feature_path,mode='validation',args=args)
    else:
        train_dataset = VideoDataset(args.json_path, args.video_path, args.csv_path,
                                     args.class_idx_path, args=args, window=args.window, mode='training')
        val_dataset = VideoDataset(args.json_path, args.video_path, args.csv_path,
                                   args.class_idx_path, args=args, window=args.window, mode='validation')

    train_loader = get_dataloader(train_dataset, True, args.workers, args.batch_size)
    val_loader = get_dataloader(val_dataset, False, args.workers, args.batch_size)

    # Set up Model
    args.manifold = MANIFOLDS[args.manifold](args.dim,args.c)
    model = VideoModel(args.manifold, args.dim, args.n_f, args.depth, args)
    model = model.to(args.device)

    # Set up Loss
    loss = get_losses(args)

    # Set up Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True, betas=(args.beta1, args.beta2))

    # Set Up Checkpoint
    checkpoint = LocalCheckpoint(
        args.checkpoint,
        include_in_all={'conf': vars(args).copy()},
        start_fresh=True
    )
    state = checkpoint.initialize({'epoch': 0, 'model': model.state_dict()})
    model.load_state_dict(state['model'])
    args.epoch_start = state['epoch']

    # Training Loop
    train_loss = 0
    val_loss = 0
    best_loss = 1e10

    for epoch in range(args.epochs):
        train_epoch_loss = 0
        for (i,(frames,label)) in tqdm(enumerate(train_loader), total=len(train_loader)):
            frames = frames.to(args.device).float()
            label = label.to(args.device)

            optimizer.zero_grad()
            preds = model(frames)

            _loss = loss(preds,label)
            _loss.backward()
            optimizer.step()
            train_epoch_loss += _loss.item()

        train_epoch_loss = train_epoch_loss/len(train_loader)
        train_loss += train_epoch_loss
        writer.add_scalar('Train/loss',train_epoch_loss,epoch)

        # Now run Validation
        with torch.no_grad():
            val_epoch_loss = 0
            for (i, (frames, label)) in tqdm(enumerate(val_loader), total=len(val_loader)):
                frames = frames.to(args.device).float()
                label = label.to(args.device)

                preds = model(frames)

                _loss = loss(preds, label)
                val_epoch_loss += _loss.item()

            val_epoch_loss = val_epoch_loss / len(val_loader)
            val_loss += val_epoch_loss

        logging.info('Epoch {}: Train Loss: {}; Val Loss: {}'.format(epoch,train_epoch_loss,val_epoch_loss))
        writer.add_scalar('Test/loss', val_epoch_loss, epoch)

        checkpoint.path = f'{args.checkpoint}.{epoch}'
        checkpoint.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'val loss': val_epoch_loss,
            'train loss': train_epoch_loss
        })

        if val_epoch_loss<=best_loss:
            logging.info('**Epoch {}: Train Loss: {}; Val Loss: {} **'.format(epoch, train_epoch_loss, val_epoch_loss))
            best_loss = val_epoch_loss
            checkpoint.path = f'{args.checkpoint}.best'
            checkpoint.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'val loss': val_epoch_loss,
                'train loss': train_epoch_loss
            })

def run_test(args):
    # Get Dataloaders
    if args.use_extracted_features:
        val_dataset = FeatureVideoDataset(args.feature_path, mode='testing', args=args)
    else:
        val_dataset = VideoDataset(args.json_path, args.video_path, args.csv_path,
                                   args.class_idx_path, args=args, window=args.window, mode='testing')

    val_loader = get_dataloader(val_dataset, False, args.workers, args.batch_size)

    # Set up Model
    args.manifold = MANIFOLDS[args.manifold](args.dim, args.c)
    model = VideoModel(args.manifold, args.dim, args.n_f, args.depth, args)
    model = model.to(args.device)

    state = torch.load(args.checkpoint+'.best')
    model.load_state_dict(state['model'])

    # Set up Loss
    loss = get_losses(args)

    with torch.no_grad():
        val_epoch_loss = 0
        for (i, (frames, label)) in tqdm(enumerate(train_loader), total=len(val_loader)):
            frames = frames.to(args.device).float()
            label = label.to(args.device)

            preds = model(frames)

            _loss = loss(preds, label)





if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    # General
    parser.add_argument('--task', type=str, default='Training', help='Features, Training, Testing')
    # Exp Logging
    parser.add_argument('--exp_name', type=str, default='Videos_Euclidean_nf_100_depth_3_lr_5e4')
    parser.add_argument('--exp_root', type=str, default='./experiments/Video_exps')
    parser.add_argument('--restore', type=bool, default=False)
    parser.add_argument('--inference', type=bool, default=False)
    # Model
    parser.add_argument('--manifold', type=str, default='euclidean', help='poincare, lorentz, euclidean')
    parser.add_argument('--loss', type=str, default='regression')
    parser.add_argument('--dim', type=int, default=5)
    parser.add_argument('--n_f', type=int, default=100)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--c', type=float, default=1)
    # Dataset
    parser.add_argument('--use_extracted_features',type=bool,default=True)
    parser.add_argument('--num_extracted_features',type=int,default=512)
    parser.add_argument('--feature_path',type=str,default='./experiments/{}_features_window_10.pt')
    parser.add_argument('--json_path', type=str, default="../hyperbolic_action-master/activity_net.v1-3.json")
    parser.add_argument('--video_path', type=str, default="/data/Activity_net/processed_jpg_64")
    parser.add_argument('--csv_path', type=str, default='./activity_net.csv')
    parser.add_argument('--class_idx_path', type=str, default='./class_indx.pkl')
    parser.add_argument('--targets_path', type=str,
                        default='/vol/medic01/users/av2514/Pycharm_projects/Topological_retrieval'
                                '/Topo_activity/experiments/activity_net_{}/activity_net_{}.best')
    parser.add_argument('--model',type=str,default='distance')
    parser.add_argument('--window', type=int, help='window of frames to represent video; -1 for all frames', default=10)
    parser.add_argument('--batch_size',type=int,default=32)
    # Optimization
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='first parameter of Adam (default: 0.9)')
    parser.add_argument('--beta2', type=float, default=0.999, help='second parameter of Adam (default: 0.900)')
    parser.add_argument('--epochs', type=int, default=200)
    # OS
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--train_threads', type=int, default=1,
                        help='Number of threads to use in training')
    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()

    # GPU utils
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # Seeds
    if args.seed == -1: args.seed = int(torch.randint(0, 2 ** 32 - 1, (1,)).item())
    print('seed', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Logging
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

    args.loss = '{}_{}'.format(args.loss,args.manifold)
    args.targets_path = args.targets_path.format(args.manifold,args.manifold)

    if args.inference:
        run_test(args)
    else:
        if args.task == 'Features':
            from feature_extraction import run_feature_extraction
            run_feature_extraction(args)
        else:
            run_train(args)
