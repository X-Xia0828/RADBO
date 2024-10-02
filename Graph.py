import networkx as nx
import torch
import heapq
import math
import numpy as np
from sklearn.mixture import GaussianMixture

def buildGraph(dataset, dim):
    Gh = nx.DiGraph()
    numOfDuel = dataset['x'].shape[0]
    for i in range(numOfDuel):
        x_id = i*2
        xx_id = i*2+1
        if dataset['y'][i] == 1.:
            Gh.add_edge(xx_id, x_id) # bad to good
        else:
            Gh.add_edge(x_id, xx_id)
    return Gh

def computeSim(dataset, dim, graphs, model, k):
    dataX = []
    dataY = []
    numOfData = dataset['x'].shape[0]*2
    kernel = model.covar_module
    for i, j in zip(dataset['x'], dataset['y']):
        dataX.append(i[:dim])
        dataY.append(j.item())
        dataX.append(i[dim:])
        dataY.append(1.0-j.item())
    tmp = 1/kernel(dataX[0].unsqueeze(0), dataX[0].unsqueeze(0)).evaluate().item()
    similarity_matrix = np.zeros((numOfData, numOfData))
    for i in range(numOfData):
        for j in range(i, numOfData):
            # Compute the similarity
            kernel_result = kernel(dataX[i].unsqueeze(0), dataX[j].unsqueeze(0)).evaluate().item()
            similarity_matrix[i, j] = kernel_result * tmp
            similarity_matrix[j, i] = similarity_matrix[i, j]
     
    distance_matrix = 1 - similarity_matrix
     
    n_clusters = k  
     
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full').fit(distance_matrix)

    labels = gmm.predict(distance_matrix)
    unique_labels = set(labels)
    
    cluster_similarities = {}
    
    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        cluster_similarity = np.mean(similarity_matrix[np.ix_(cluster_indices, cluster_indices)])
        cluster_similarities[label] = cluster_similarity
    best_cluster_label = max(cluster_similarities, key=cluster_similarities.get)
    best_cluster_points = np.where(labels == best_cluster_label)[0]
    set1 = []
    set2 = []
    for id0 in best_cluster_points:
        id1 = id0 + (-1)**id0
        if id1 in best_cluster_points:
            continue
        if dataY[id0] == 1.:
            set1.append(id1)
        else:
            set2.append(id1)
    for id1 in set1:
        for id2 in best_cluster_points:
            graphs.add_edge(id1, id2)
    for id1 in best_cluster_points:
        for id2 in set2:
            graphs.add_edge(id1, id2)
    return graphs

def createDataset(dataset, dim, graph):
    print(graph)
    all_nodes_id = list(graph.nodes)
    x = []
    y = []
    for node_id in all_nodes_id:   
        if node_id%2 == 0: 
            node = dataset['x'][node_id//2][:dim] 
        else:
            node = dataset['x'][node_id//2][dim:] 
        reachable_nodes_id = nx.descendants(graph, node_id)
        for nd_id in reachable_nodes_id:
            if nd_id//2 == node_id//2: #skip owned points
                continue
            if nd_id%2 == 0: 
                nd = dataset['x'][nd_id//2][:dim] 
            else:
                nd = dataset['x'][nd_id//2][dim:] 
            x.append(torch.cat([node, nd], 0).tolist())
            y.append(0.)
            x.append(torch.cat([nd, node], 0).tolist())
            y.append(1.)
    x = torch.tensor(x)
    y = torch.tensor(y).reshape(-1, 1)
    new_dataset = {'x': x, 'y' : y}
    return new_dataset
    

def dealDataset(dataset, dim):
    numOfdata = dataset['x'].shape[0]
    data_trans = []
    prefer_trans = []
    data_single = []
    prefer_single = []
    for i, j in zip(dataset['x'], dataset['y']):
        data_trans.append(torch.cat([i[dim:], i[:dim]], 0).tolist())
        if j.item() == 1.:
            prefer_trans.append(0.)
        else:
            prefer_trans.append(1.)
        data_single.append(i[:dim].tolist())
        data_single.append(i[dim:].tolist())
        if j.item() == 1.:
            prefer_single.append(1.)
            prefer_single.append(0.)
        else:
            prefer_single.append(0.)
            prefer_single.append(1.)
    data_trans = torch.tensor(data_trans).reshape(numOfdata, -1)
    prefer_trans = torch.tensor(prefer_trans).reshape(-1, 1)

    data_single = torch.tensor(data_single).reshape(numOfdata*2, -1)
    prefer_single = torch.tensor(prefer_single).reshape(-1, 1)

    train_x = torch.tensor(torch.cat([dataset['x'], data_trans], 0))
    train_y = torch.tensor(torch.cat([dataset['y'], prefer_trans], 0))
    dataset_old = {'x': train_x, 'y': train_y}
    dataset_single = {'x': data_single, 'y': prefer_single}
    return dataset_old, dataset_single
