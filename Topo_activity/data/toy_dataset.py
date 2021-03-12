from sklearn.datasets import make_swiss_roll
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.neighbors import kneighbors_graph
from collections import defaultdict

np.random.seed(0)
output_dir = ''
dict_to_save = defaultdict()

n_samples = 20000
noise = 0.05
X, _ = make_swiss_roll(n_samples, noise=noise)

ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(X)
label = ward.labels_

dict_to_save['data'] = X
dict_to_save['labels'] = label

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
for l in np.unique(label):
    ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2],
               color=plt.cm.jet(float(l) / np.max(label + 1)),
               s=20, edgecolor='k')
plt.title('No connectivity, Anglomerative Clustering')
plt.savefig()
plt.show()

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -50)
for l in np.unique(label):
    ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2],
               color=plt.cm.jet(float(l) / np.max(label + 1)),
               s=20, edgecolor='k')
plt.title('No connectivity, Anglomerative Clustering')
plt.show()


connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
ward = AgglomerativeClustering(n_clusters=6, connectivity=connectivity,
                               linkage='ward').fit(X)
label = ward.labels_


fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
for l in np.unique(label):
    ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2],
               color=plt.cm.jet(float(l) / np.max(label + 1)),
               s=20, edgecolor='k')
plt.title('With connectivity constraints ')
plt.show()

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -50)
for l in np.unique(label):
    ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2],
               color=plt.cm.jet(float(l) / np.max(label + 1)),
               s=20, edgecolor='k')
plt.title('With connectivity constraints ')
plt.show()

