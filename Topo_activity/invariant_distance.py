import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import os
plt.rcParams.update({'font.size': 14})

file_root = '/vol/medic01/users/av2514/Pycharm_projects/Topological_retrieval/Topo_activity/experiments/diagrams'
# file_list = ['activity_net_euclidean_cosine.npz','activity_net_euclidean.npz']
# file_list = ['activity_net_euclidean_expmap_poincare.npz','activity_net_poincare.npz']
# file_list = ['Activity_net_no_embedding_euclidean.npz','activity_net_euclidean.npz']
# file_list = ['Mammals_no_embedding_euclidean.npz','mammals_euclidean.npz']
# file_list = ['mammals_euclidean_cosine.npz', 'mammals_euclidean.npz']
file_list_master = [('activity_net_euclidean_cosine.npz','activity_net_euclidean.npz'),
    ('activity_net_euclidean_expmap_poincare.npz','activity_net_poincare.npz'),
    ('mammals_euclidean_cosine.npz', 'mammals_euclidean.npz'),
    ('mammals_euclidean_expmap_poincare.npz','mammals_poincare.npz')]

### Read files
datasets = ['Activity Net','Activity Net','Mammals','Mammals']
metrics = ['Euclidean','Poincare','Euclidean','Poincare']
multipliers = [(1,1),(0.5,2),(0.6,0.7), (0.5,0.55)]
dict={}
for jj, file_list in enumerate(file_list_master):
    dist_list = [np.load(os.path.join(file_root,u))['arr_0'] for u in file_list]

    ### Set dilation interval according to max persistence and mean persistence
    X = dist_list[0]
    meanPersisX = np.mean(X[0, :, 1] - X[0, :, 0])
    maxPersisX = np.max(X[0, :, 1] - X[0, :, 0])

    Y = dist_list[1]
    meanPersisY = np.mean(Y[0, :, 1] - Y[0, :, 0])
    maxPersisY = np.max(Y[0, :, 1] - Y[0, :, 0])

    dila0 = meanPersisY / meanPersisX
    dila1 = maxPersisY / maxPersisX

    dilaMin = min(dila0, dila1)
    dilaMax = max(dila0, dila1)

    ### The parameters are set by hand
    # for activity net euclidean, dilaMin, dilaMax is fine
    # for activity poincare, 0.5*dilaMin, 2*dilaMax is fine
    # for mammal euclidean, 0.6*dilaMin, 0.7*dilaMax is fine
    # for mammal poincare, 0.5*dilaMin, 0.55*dilaMax is fine
    dilaInterv = np.linspace( multipliers[jj][0]*dilaMin, multipliers[jj][1]*dilaMax, 20)

    ### append identical dilation
    dilaInterv = np.append(dilaInterv, 1)

    ### separate dimension 0,1,2
    Xind0 = np.nonzero(X[0, :, 2] == 0)
    Xind1 = np.nonzero(X[0, :, 2] == 1)
    Xind2 = np.nonzero(X[0, :, 2] == 2)
    Yind0 = np.nonzero(Y[0, :, 2] == 0)
    Yind1 = np.nonzero(Y[0, :, 2] == 1)
    Yind2 = np.nonzero(Y[0, :, 2] == 2)
    Xarray = np.reshape(X[0, :, 0:2], (-1, 2))
    Yarray = np.reshape(Y[0, :, 0:2], (-1, 2))
    Yper0 = Yarray[Yind0[0], :]
    Yper1 = Yarray[Yind1[0], :]
    Yper2 = Yarray[Yind2[0], :]

    bottleneckDistance = []

    # compute bottleneck distance
    for i in range(0, len(dilaInterv)):
        cX = Xarray
        cX = dilaInterv[i] * cX
        # dimension 0
        cXper0 = cX[Xind0[0], :]
        cDistance0 = gd.bottleneck_distance(cXper0, Yper0)
        # dimension 1
        cXper1 = cX[Xind1[0], :]
        cDistance1 = gd.bottleneck_distance(cXper1, Yper1)
        # dimension 2
        cXper2 = cX[Xind2[0], :]
        cDistance2 = gd.bottleneck_distance(cXper2, Yper2)
        # bottleneck distance
        cDistance = max(cDistance0, cDistance1, cDistance2)
        bottleneckDistance.append(cDistance)

    ### plot

    dataset = datasets[jj]
    metric = metrics[jj]
    minBottleneckDistance = min(bottleneckDistance)
    argminbot = np.argmin(bottleneckDistance)

    dict['{}_{}'.format(dataset,metric)] = {'dilation_interval': dilaInterv[0:len(dilaInterv) - 1],
                                            'Different Dilations': bottleneckDistance[0:len(bottleneckDistance) - 1],
                                            'No Dilation':bottleneckDistance[-1],
                                            'Optimal Dilation':minBottleneckDistance,
                                            'whereOptimal':dilaInterv[argminbot]}

    plt.figure()

    label = ['Different Dilations', 'No Dilation', 'Optimal Dilation']
    plt.xlabel("Dilation Parameters")
    plt.ylabel("Bottleneck Distances")
    plt.plot(dilaInterv[0:len(dilaInterv) - 1], bottleneckDistance[0:len(bottleneckDistance) - 1])
    # the last element corresponds to no dilation
    plt.axhline(y=bottleneckDistance[-1], color='r', linestyle='-.')
    plt.axhline(y=minBottleneckDistance, color='r', linestyle='-')
    plt.legend(label, loc='best')
    # plt.title('{} - {} Metrics - {}'.format(dataset,metric,expl))
    plt.savefig('./experiments/imgdump/bottleneck_{}_{}.pdf'.format(dataset,metric),format='pdf')
    plt.savefig('./experiments/imgdump/bottleneck_{}_{}.eps'.format(dataset,metric),format='eps')

with open('numbers.csv', 'w') as f:
    for key in dict.keys():
        f.write("%s,%s\n"%(key,dict[key]))