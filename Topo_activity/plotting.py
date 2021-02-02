import torch
import numpy
import os
import numpy as np
import networkx as nx
from data.dataloader import TreeDataset
from plotly.graph_objs import *
import chart_studio.plotly as py
import pandas as pd

width=800
height=800
axis=dict(showline=False, # hide axis line, grid, ticklabels and  title
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title=''
          )

layout=Layout(title={ 'text': "WordNet Mammals",
                      'xanchor': 'left'
                      },
    font= dict(size=12),
    showlegend=False,
    autosize=False,
    width=width,
    height=height,
    xaxis=layout.XAxis(axis),
    yaxis=layout.YAxis(axis),
    margin=layout.Margin(
        l=40,
        r=40,
        b=85,
        t=100,
    ),
    hovermode='closest',
    annotations=[
           dict(
           showarrow=False,
            text='',
            xref='paper',
            yref='paper',
            x=0,
            y=-0.1,
            xanchor='left',
            yanchor='bottom',
            font=dict(
            size=14
            )
            )
        ]
    )

json_path = "../hyperbolic_action-master/activity_net.v1-3.json"
tree_dataset = TreeDataset(json_path)

nodeids = [item['nodeId'] for item in tree_dataset.taxonomy]
nodeids_par = [item['parentId'] for item in tree_dataset.taxonomy]
labels = [item['nodeName'] for item in tree_dataset.taxonomy]
unique_ids = np.unique(nodeids)

edges = [(item['nodeId'],item['parentId']) for item in tree_dataset.taxonomy]

V=range(len(nodeids))
g=nx.Graph()

g.add_nodes_from(nodeids)
g.add_edges_from(edges)

pos=nx.fruchterman_reingold_layout(g)

Xv=[pos[k][0] for k in unique_ids]
Yv=[pos[k][1] for k in unique_ids]
a = np.zeros((len(Xv), 2))
for i in range(len(Xv)):
    a[i, 0] = Xv[i]
    a[i, 1] = Yv[i]

Xed=[]
Yed=[]
for edge in edges:
    Xed+=[pos[edge[0]][0],pos[edge[1]][0], None]
    Yed+=[pos[edge[0]][1],pos[edge[1]][1], None]

trace3=Scatter(x=Xed,
               y=Yed,
               mode='lines',
               line=dict(color='rgb(210,210,210)', width=1),
               hoverinfo='none'
               )
trace4=Scatter(x=Xv,
               y=Yv,
               mode='markers',
               name='net',
               marker=dict(symbol='circle-dot',
                             size=5,
                             color='#6959CD',
                             line=dict(color='rgb(50,50,50)', width=0.5)
                             ),
               text=labels,
               hoverinfo='text'
               )
data1=[trace3, trace4]
annot="Activity Net"

fig1=Figure(data=data1)#, layout=layout)
fig1.write_image('./experiments/imgdump/activity_net_network.pdf')
# fig1['layout']['annotations'][0]['text']=annot
# py.iplot(fig1, filename='Coautorship-network-nx')


csv_path = '/vol/medic01/users/av2514/Pycharm_projects/Topological_retrieval/others/poincare-embeddings-master/wordnet/mammal_closure.csv'

db = pd.read_csv(csv_path)
unique_ids = np.unique(db['id1'])
unique_ids_2 = np.unique(db['id2'])

all_uniques = list(set(unique_ids).union(unique_ids_2))
edges = []
for row in db.iterrows():
    edges.append((all_uniques.index(row[1]['id1']),all_uniques.index(row[1]['id2'])))


V=range(len(all_uniques))
g=nx.Graph()

g.add_nodes_from(V)
g.add_edges_from(edges)

pos=nx.fruchterman_reingold_layout(g)

Xv=[pos[k][0] for k in V]
Yv=[pos[k][1] for k in V]
Xed=[]
Yed=[]
for edge in edges:
    Xed+=[pos[edge[0]][0],pos[edge[1]][0], None]
    Yed+=[pos[edge[0]][1],pos[edge[1]][1], None]

trace3=Scatter(x=Xed,
               y=Yed,
               mode='lines',
               line=dict(color='rgb(210,210,210)', width=1),
               hoverinfo='none'
               )
trace4=Scatter(x=Xv,
               y=Yv,
               mode='markers',
               name='net',
               marker=dict(symbol='circle-dot',
                             size=5,
                             color='#6959CD',
                             line=dict(color='rgb(50,50,50)', width=0.5)
                             ),
               text=unique_ids,
               hoverinfo='text'
               )
data1=[trace3, trace4]
annot="Mammals"

fig1=Figure(data=data1, layout=layout)
fig1.write_image('./experiments/imgdump/wordnet_mammals_network.pdf')