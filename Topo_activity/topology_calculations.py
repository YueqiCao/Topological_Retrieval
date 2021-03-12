import torch
import os
import numpy as np
import argparse
from model import get_model
from gtda.homology import VietorisRipsPersistence
from data.dataloader import TreeDataset
from gtda.plotting import plot_diagram
import manifolds
import manifolds.pmath as pmath
import gudhi
import gtda



def get_targets(args, N, targets_path):
    args.sparse = True
    args.margin=0.1
    model = get_model(args, N)
    state = torch.load(targets_path)
    model.load_state_dict(state['model'])
    return model

def get_diagrams(args,embs,metric,plot=True,plot_name='euclidean'):
    VR = VietorisRipsPersistence(homology_dimensions=[0, 1, 2], n_jobs=1, metric=metric)
    embs = embs[np.newaxis,...]
    diagrams = VR.fit_transform(embs)
    np.savez('./experiments/diagrams/{}.npz'.format(plot_name),diagrams)
    if plot:
        i = 0
        plt = plot_diagram(diagrams[i])
        plt.write_image('./experiments/imgdump/{}.pdf'.format(plot_name))

    return diagrams


def topology_investigation(args):
    dataset = TreeDataset(args.json_path, negs=args.negs,
                          batch_size=1,
                          ndproc=args.ndproc,
                          burnin=1,
                          dampening=args.dampening,
                          neg_multiplier=args.neg_multiplier)

    data = dataset.get_graph_dataset()

    args.manifold = 'poincare'
    poincare_model = get_targets(args, args.N,os.path.join(args.targets_path_root,'{}_poincare/{}_poincare.pth.best'.format(args.dataset,args.dataset)))
    args.manifold = 'lorentz'
    lorentz_model = get_targets(args, args.N,os.path.join(args.targets_path_root,'{}_lorentz/{}_lorentz.pth.best'.format(args.dataset,args.dataset)))
    args.manifold = 'euclidean'
    euclidean_model = get_targets(args, args.N,os.path.join(args.targets_path_root,'{}_euclidean/{}_euclidean.pth.best'.format(args.dataset,args.dataset)))

    # get Euclidean VR persistence
    print('Calculating Euclidean ...')
    euclidean_embs = euclidean_model.lt.weight.data.cpu().numpy()
    eucl_diagram = get_diagrams(args, euclidean_embs, metric='euclidean', plot=True, plot_name='{}_euclidean'.format(args.dataset))
    eucl_cosine_diagram = get_diagrams(args, euclidean_embs, metric='cosine', plot=True, plot_name='{}_euclidean_cosine'.format(args.dataset))

    # Get Lorentz VR Persistence
    print('Calculating Lorentz ...')
    lorentz_manifold =  manifolds.LorentzManifold
    lorentz_embs = lorentz_model.lt.weight.data.cpu().numpy()
    lorentz_diagram = get_diagrams(args, lorentz_embs, metric=lorentz_manifold.distance_wrapper, plot=True, plot_name='{}_lorentz'.format(args.dataset))

    # Get Poincare VR Persistence
    print('Calculating Poincare ...')
    poincare_manifold = manifolds.PoincareManifold
    poincare_embs = poincare_model.lt.weight.data.cpu().numpy()
    poincare_diagram = get_diagrams(args, poincare_embs, metric=poincare_manifold.distance_wrapper, plot=True, plot_name='{}_poincare'.format(args.dataset))

    # Get expmap Euclidean -> Poincare VR Persistence
    exp_eucl = pmath.expmap0(euclidean_model.lt.weight).data.cpu().numpy()
    eucl_exp_map_diagram = get_diagrams(args, exp_eucl, metric=poincare_manifold.distance_wrapper, plot=True, plot_name='{}_euclidean_expmap_poincare'.format(args.dataset))

    # Euclidaen bottleneck
    a = [ (item[0],item[1]) for item in eucl_diagram[0]]
    b = [ (item[0],item[1]) for item in eucl_cosine_diagram[0]]
    eucl_dist = gtda.externals.bottleneck_distance(a,b)
    print('Euclidean Bottleneck of Euclidean distance: {}'.format(eucl_dist))

    eucl_dist = gudhi.bottleneck_distance(np.array(a), np.array(b))
    print('Euclidean Bottleneck of Euclidean distance: {}'.format(eucl_dist))

    # Poincare bottleneck
    a = [(item[0], item[1]) for item in poincare_diagram[0]]
    b = [(item[0], item[1]) for item in eucl_exp_map_diagram[0]]
    poinc_dist =gudhi.bottleneck_distance(np.array(a), np.array(b))
    print('Poincare Bottleneck of Poincare distance: {}'.format(poinc_dist))




if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model',type=str,default='distance')
    # parser.add_argument('--dataset',type=str,default='activity_net')
    parser.add_argument('--dataset',type=str,default='mammals')
    parser.add_argument('--N',type=int,default=1180)
    # parser.add_argument('--N',type=int,default=273)
    parser.add_argument('--dim', type=int, default=5)
    parser.add_argument('--targets_path_root', type=str,
                        default='/vol/medic01/users/av2514/Pycharm_projects/Topological_retrieval'
                                '/Topo_activity/experiments/')
    # Dataset
    parser.add_argument('--json_path', type=str, default="../hyperbolic_action-master/activity_net.v1-3.json")
    parser.add_argument('--negs', type=int, default=50, help='negative samples')
    parser.add_argument('--ndproc', type=int, default=4, help='Number of data loading processes')
    parser.add_argument('--dampening', type=float, default=0.75, help='Sample dampening during burnin')
    parser.add_argument('--neg_multiplier', type=float, default=1.0)
    parser.add_argument('--manifold', type=str, default='poincare', help='poincare, lorentz, euclidean')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    if args.seed == -1: args.seed = int(torch.randint(0, 2 ** 32 - 1, (1,)).item())
    print('seed', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    topology_investigation(args)
