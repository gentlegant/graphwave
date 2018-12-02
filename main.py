# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



import graphwave
from graphwave.graphwave import *

from graphwave.shapes import build_graph
from graphwave.shapes.shapes import plot_networkx

np.random.seed(123)
#--------------------------------------------------------------------------------------
# 1- Start by defining our favorite regular structure

width_basis = 45

add_edges = 5

################################### EXAMPLE TO BUILD A SIMPLE REGULAR STRUCTURE ##########
## REGULAR STRUCTURE: the most simple structure:  basis + n small patterns of a single type

### 1. Choose the basis (cycle, torus or chain)
basis_type = "cycle"

### 2. Add the shapes
n_shapes = 5  ## numbers of shapes to add
#shape=["fan",6] ## shapes and their associated required parameters  (nb of edges for the star, etc)
#shape=["star",6]
list_shapes = [["house"]] * n_shapes

### 3. Give a name to the graph
identifier = 'AA'  ## just a name to distinguish between different trials

name_graph = 'houses'+ identifier

sb.set_style('white')

### 4. Pass all these parameters to the Graph Structure

# G, communities, _ , role_id = build_graph.build_structure(width_basis, basis_type, list_shapes, start=0,
#                                        add_random_edges=add_edges, plot=True,
#                                        savefig=False)

#用我的function来加载图
G=build_graph.get_graph_from_path("C:\\Users\\james\\Desktop\\graphwave\\graph\\europe-airports.edgelist")
role_id=build_graph.get_role_from_path("C:\\Users\\james\\Desktop\\graphwave\\graph\\labels-europe-airports.txt")
plot_networkx(G,role_id)

print "number of edges",G.number_of_edges()
print "number of nodes",G.number_of_nodes()
print "matrix shape",nx.adj_matrix(G).shape


chi, heat_print, taus = graphwave_alg(G, np.linspace(0,100,25), taus='auto', verbose=True)

nb_clust = len(np.unique(role_id))

print (u'共有'+str(nb_clust)+u'种角色')

pca = PCA(n_components=5)

trans_data = pca.fit_transform(StandardScaler().fit_transform(chi))
print(chi.shape)
trans_data = pca.fit_transform(chi)

km = KMeans(n_clusters=nb_clust)

km.fit(trans_data)

labels_pred = km.labels_
#有多少种角色，这就被 聚类成多少种
######## Params for plotting
cmapx = plt.get_cmap('rainbow')
x = np.linspace(0, 1, nb_clust + 1)
col = [cmapx(xx) for xx in x]
markers = {0: '*', 1: '.', 2: ',', 3: 'o', 4: 'v', 5: '^', 6: '<', 7: '>', 8: 3, 9: 'd', 10: '+', 11: 'x', 12: 'D',
           13: '|', 14: '_', 15: 4, 16: 0, 17: 1, 18: 2, 19: 6, 20: 7}

for c in np.unique(role_id):
    indc = [i for i, x in enumerate(role_id) if x == c]   #role_id 相同的  下标
    print(c)
    print(np.var(trans_data[indc, 0]))
    print(np.var(trans_data[indc, 1]))
    plt.scatter(trans_data[indc, 0], trans_data[indc, 1],
                c=np.array(col)[list(np.array(labels_pred)[indc])],    #把 当前role_id的所有labels_pred列出来，选取不同颜色
                s=300)

labels = role_id
for label, x, y in zip(labels, trans_data[:, 0], trans_data[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
plt.show()
