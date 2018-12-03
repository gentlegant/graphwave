# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import Graphwave
import Graphwave.graphwave
from Graphwave import shapes
from Graphwave.graphwave import graphwave_alg
from Graphwave.shapes import build_graph
from Graphwave.shapes.shapes import plot_networkx

np.random.seed(123)
#--------------------------------------------------------------------------------------
# 1- Start by defining our favorite regular structure

width_basis = 25

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


#chi是根据特征函数的embedding，chi1是根据myfunction 的embedding
emb1,emb2, heat_print, taus = graphwave_alg(G, np.linspace(0,100,25), taus=[0.8,1], verbose=True)


tPlot, axes = plt.subplots(
        nrows=len(np.unique(role_id)), ncols=4, sharex=True, sharey=False,        
        )
tPlot.suptitle('能量值', fontsize=20)

for index,c in  enumerate(np.unique(role_id)):
    indc = [i for i, x in enumerate(role_id) if x == c] 
    a=heat_print[0].getA()[indc[0]]
    b=heat_print[0].getA()[indc[1]]
    axes[index][0].bar(range(len(a)),a)
    axes[index][1].bar(range(len(b)),b)
    c=heat_print[1].getA()[indc[0]]
    d=heat_print[1].getA()[indc[1]]
    axes[index][2].bar(range(len(c)),d)
    axes[index][3].bar(range(len(c)),c)


plt.show()



nb_clust = len(np.unique(role_id))

print (u'共有'+str(nb_clust)+u'种角色')

pca = PCA(n_components=5)

#trans_data = pca.fit_transform(StandardScaler().fit_transform(chi))


trans_data = pca.fit_transform(emb1)


print("chi.shape:")
print(emb1.shape)
print("chi1.shape:")
print(emb2.shape)


trans_data = pca.fit_transform(emb1)
trans_data1 = pca.fit_transform(emb2)

km = KMeans(n_clusters=nb_clust)

km.fit(trans_data)

labels_pred = km.labels_

km.fit(trans_data1)

labels_pred1=km.labels_
#有多少种角色，这就被 聚类成多少种
######## Params for plotting
cmapx = plt.get_cmap('rainbow')
x = np.linspace(0, 1, nb_clust + 1)
col = [cmapx(xx) for xx in x]
# markers = {0: '*', 1: '.', 2: ',', 3: 'o', 4: 'v', 5: '^', 6: '<', 7: '>', 8: 3, 9: 'd', 10: '+', 11: 'x', 12: 'D',
#            13: '|', 14: '_', 15: 4, 16: 0, 17: 1, 18: 2, 19: 6, 20: 7}

tPlot, axes = plt.subplots(
        nrows=1, ncols=2, sharex=True, sharey=False,        
        )


for c in np.unique(role_id):
    indc = [i for i, x in enumerate(role_id) if x == c]   #role_id 相同的  下标
   
 
    print(u"根据character 类内 第一维 方差")
    print(np.var(trans_data[indc, 0]))
    print(u"根据我的模型   类内 第一维 方差")
    print(np.var(trans_data1[indc, 0]))
    axes[0].scatter(trans_data[indc, 0], trans_data[indc, 1],
                c=np.array(col)[list(np.array(labels_pred)[indc])],
                 #把 当前role_id的所有labels_pred列出来，选取不同颜色
                s=300)
    axes[1].scatter(trans_data1[indc, 0], trans_data1[indc, 1],
                c=np.array(col)[list(np.array(labels_pred1)[indc])],
                 #把 当前role_id的所有labels_pred列出来，选取不同颜色
                s=300)

#标记本身的类别，（使用数字）
for label, x, y in zip(role_id, trans_data[:, 0], trans_data[:, 1]):  
    axes[0].annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
for label, x, y in zip(role_id, trans_data1[:, 0], trans_data1[:, 1]):  
    axes[1].annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
 
plt.show()
