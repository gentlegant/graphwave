#coding=utf-8 
import copy
import math
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy as sc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from characteristic_functions import (charac_function,
                                      charac_function_multiscale)
from utils.graph_tools import laplacian




TAUS = [1, 10, 25, 50]
ORDER = 30
PROC = 'exact'
ETA_MAX = 0.95
ETA_MIN = 0.80
NB_FILTERS = 2

base=10



def compute_cheb_coeff(scale, order):
    coeffs = [(-scale)**k * 1.0 / math.factorial(k) for k in range(order + 1)]
    return coeffs


def compute_cheb_coeff_basis(scale, order):
    xx = np.array([np.cos((2 * i - 1) * 1.0 / (2 * order) * math.pi)
                   for i in range(1, order + 1)])
    basis = [np.ones((1, order)), np.array(xx)]
    for k in range(order + 1-2):
        basis.append(2* np.multiply(xx, basis[-1]) - basis[-2])
    basis = np.vstack(basis)
    f = np.exp(-scale * (xx + 1))
    products = np.einsum("j,ij->ij", f, basis)
    coeffs = 2.0 / order * products.sum(1)
    coeffs[0] = coeffs[0] / 2
    return list(coeffs)


def heat_diffusion_ind(graph, taus=TAUS, order = ORDER, proc = PROC):
    '''
    This method computes the heat diffusion waves for each of the nodes
    INPUT:
    -----------------------
    graph    :    Graph (etworkx)
    taus     :    list of scales for the wavelets. The higher the tau,
                  the better the spread of the heat over the graph
    order    :    order of the polynomial approximation
    proc     :    which procedure to compute the signatures (approximate == that
                  is, with Chebychev approx -- or exact)

    OUTPUT:
    -----------------------
    heat     :     tensor of length  len(tau) x n_nodes x n_nodes
                   where heat[tau,:,u] is the wavelet for node u
                   at scale tau
    taus     :     the associated scales
    '''
    # Compute Laplacian
    a = nx.adjacency_matrix(graph)
   
    n_nodes, _ = a.shape
    
    #thres = np.vectorize(lambda x : x if x > 1e-4 * 1.0 / n_nodes else 0)
    #返回的是一个 参数 为 列表的函数

    lap = laplacian(a)
    
    

    n_filters = len(taus)
    # if proc == 'exact':
        ### Compute the exact signature
    lamb, U = np.linalg.eigh(lap.todense())

    heat = {}
    for i in range(n_filters):
        #type of K :  <class 'numpy.matrixlib.defmatrix.matrix'>
        K = U.dot(np.diagflat(np.exp(-taus[i] * lamb).flatten())).dot(U.T)

        
        #heat[i]=sc.sparse.csc_matrix(K)
        heat[i]=sc.matrix(K)
            
    # else:
    #     heat = {i: sc.sparse.csc_matrix((n_nodes, n_nodes)) for i in range(n_filters) } 
    #     #this create a dict with n_filters empyt sparse matrix
    #     monome = {0: sc.sparse.eye(n_nodes), 1: lap - sc.sparse.eye(n_nodes)}
    #     for k in range(2, order + 1):
    #          monome[k] = 2 * (lap - sc.sparse.eye(n_nodes)).dot(monome[k-1]) - monome[k - 2]
    #     for i in range(n_filters):
    #         coeffs = compute_cheb_coeff_basis(taus[i], order)
    #         heat[i] = sc.sum([ coeffs[k] * monome[k] for k in range(0, order + 1)])
    #         temp = thres(heat[i].A) # cleans up the small coefficients
            
    #         heat[i] = sc.sparse.csc_matrix(temp)
 
    return heat, taus    #there is something wrong with this function ,this function may return heat with different type by different proc.


def myfunc(heat):
    tem=[]
    for matrix in heat.values():
        print("----------------------------------")
        matrix.sort()  #如果sort没有指定axis ，默认按最后一个 axis排序，在这就是按行排序，或者说排序每行
     
        tem.append(matrix)

    return np.column_stack(tem)  #横着组合
    

def myfunc2(heat):
    
   
    heat[0].sort()  
        
    return heat[0]
#对一个高维度数组打包装箱
def bins(length,p,func):
   
    res=[]
    start=0
    rest=length

    while rest!=0:
        l=int(rest*p) if int(rest*p)!=0 else 1 
        rest-=l
        arr=np.zeros((length))
        if(func=="sum"):
            arr[start:start+l]=1
        elif (func=="mean"):
            arr[start:start+l]=1/l
        res.append(arr)
        start+=l
    
    return np.column_stack(res)

def myfunc3(heat):
    tmp2=[]
    
    bins_tran=bins(heat[0].shape[0],0.5,"sum")
    for matrix in heat.values():
        cp=matrix.copy()
        cp.sort()
        
        
        tmp2.append(cp.dot(bins_tran))
    return np.column_stack(tmp2)




def myfunc4(heat):
    tmp2=[]
    for mat in heat.values():
        
        lildata=mat.toarray()
        lildata.sort(1)
        tmp=[]
        for row in lildata:
            tmp.append(bins(row,0.1,np.sum))
        tmp2.append(np.row_stack(tmp))
    return np.column_stack(tmp2)




def graphwave_alg(graph, time_pnts, taus= 'auto', 
              verbose=False, approximate_lambda=True,
              order=ORDER, proc=PROC, nb_filters=NB_FILTERS,
              **kwargs):
    '''
    wrapper function for computing the structural signatures using GraphWave
    INPUT
    --------------------------------------------------------------------------------------
    graph             :   nx Graph
    time_pnts         :   time points at which to evaluate the characteristic function
    taus              :   list of scales that we are interested in. Alternatively,
                          'auto' for the automatic version of GraphWave
    verbose           :   the algorithm prints some of the hidden parameters
                          as it goes along
    approximate_lambda:   (boolean) should the range oflambda be approximated or
                          computed?
    proc              :   which procedure to compute the signatures (approximate == that
                          is, with Chebychev approx -- or exact)
    nb_filters        :   nuber of taus that we require if  taus=='auto'
    OUTPUT
    --------------------------------------------------------------------------------------
    chi               :  embedding of the function in Euclidean space
    heat_print        :  returns the actual embeddings of the nodes
    taus              :  returns the list of scales used.
    '''
   
    if taus == 'auto':
        if approximate_lambda is not True:
            a = nx.adjacency_matrix(graph)
            lap = laplacian(a)
            b=sc.sparse.linalg.eigsh(lap, 2,  which='SM',return_eigenvectors=False)
         
            
            try:
                l1 = np.sort(sc.sparse.linalg.eigsh(lap, 2,  which='SM',return_eigenvectors=False))[1]
                #return_eigenvectors=False means this only return eigenvalue, so  l1 is the secend samllest eigenvalue.
            except:
                l1 = np.sort(sc.sparse.linalg.eigsh(lap, 5,  which='SM',return_eigenvectors=False))[1]
        else:
            l1 = 1.0 / graph.number_of_nodes()
        smax = -np.log(ETA_MIN) * np.sqrt( 0.5 / l1)
        smin = -np.log(ETA_MAX) * np.sqrt( 0.5 / l1)
        taus = np.linspace(smin, smax, nb_filters)
    #这是模型的第一步，不可以省略
    heat_print, _ = heat_diffusion_ind(graph, list(taus), order=order, proc = proc)
  

    chi = charac_function_multiscale(heat_print, time_pnts)
    chi1=myfunc3(heat_print)

    
    
    return chi,chi1, heat_print, taus

