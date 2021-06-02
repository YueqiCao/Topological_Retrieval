import sys
sys.path.append(".")
sys.path.append("..")
from sklearn.datasets import make_swiss_roll
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.neighbors import kneighbors_graph
from collections import defaultdict
import pickle
import os
from utils import pickle_object


np.random.seed(0)
output_dir = '/vol/medic01/users/av2514/Pycharm_projects/Topological_retrieval/Data'
dict_to_save = defaultdict()

n_samples = 20000
noise = 0.05
X, _ = make_swiss_roll(n_samples, noise=noise)

ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(X)
label = ward.labels_

dict_to_save['data'] = X
dict_to_save['labels'] = label

dict_to_save['n_samples'] = n_samples
dict_to_save['noise'] = noise
dict_to_save['seed'] = 0
dict_to_save['clustering'] = 'AgglomerativeClustering'
dict_to_save['n_cluters'] = 6
dict_to_save['linkage'] = 'ward'
dict_to_save['connectivity'] = 'None'
dict_to_save['n_neighbors'] = 0

output_name = 'swiss_roll_n_samples_{}'.format(n_samples)
outpath = os.path.join(output_dir,output_name+'.pkl')

pickle_object(dict_to_save, outpath)

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
for l in np.unique(label):
    ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2],
               color=plt.cm.jet(float(l) / np.max(label + 1)),
               s=20, edgecolor='k')
plt.title('No connectivity, Anglomerative Clustering')
plt.savefig(os.path.join(output_dir,output_name+'_1.pdf'),format='pdf')
# plt.show()

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -50)
for l in np.unique(label):
    ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2],
               color=plt.cm.jet(float(l) / np.max(label + 1)),
               s=20, edgecolor='k')
plt.title('No connectivity, Anglomerative Clustering')
plt.savefig(os.path.join(output_dir,output_name+'_2.pdf'),format='pdf')

# plt.show()


connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
ward = AgglomerativeClustering(n_clusters=6, connectivity=connectivity,
                               linkage='ward').fit(X)
label = ward.labels_
dict_to_save = defaultdict()

dict_to_save['data'] = X
dict_to_save['labels'] = label

dict_to_save['n_samples'] = n_samples
dict_to_save['noise'] = noise
dict_to_save['seed'] = 0
dict_to_save['clustering'] = 'AgglomerativeClustering'
dict_to_save['n_cluters'] = 6
dict_to_save['linkage'] = 'ward'
dict_to_save['connectivity'] = 'knn'
dict_to_save['n_neighbors'] = 10

output_name = 'swiss_roll_n_samples_{}_with_connectivity'.format(n_samples)
outpath = os.path.join(output_dir,output_name+'.pkl')

pickle_object(dict_to_save, outpath)

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
for l in np.unique(label):
    ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2],
               color=plt.cm.jet(float(l) / np.max(label + 1)),
               s=20, edgecolor='k')
plt.title('With connectivity constraints ')
plt.savefig(os.path.join(output_dir,output_name+'_1.pdf'),format='pdf')

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -50)
for l in np.unique(label):
    ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2],
               color=plt.cm.jet(float(l) / np.max(label + 1)),
               s=20, edgecolor='k')
plt.title('With connectivity constraints ')
plt.savefig(os.path.join(output_dir,output_name+'_2.pdf'),format='pdf')


