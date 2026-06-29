# Code that defines functions for visualizing the data

import torch
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx

def visualize_snapshot_cell(data_list, idx, name):
    """
    data_list: list of input data that is of shape (snapshot, [Torch data object])
    idx: list of indices within data_list that you want to visualize
    """
    for i in idx:
        cell_ids = data_list[i].voronoi_equivalent

        plt.figure()
        plt.imshow(cell_ids)
        plt.savefig(f'training_results_batch3_[norm]/{name}_ssCell_idx{i}.png', dpi=200)
        plt.close()

def visualize_snapshot_nucleus(data_list, idx, name):
    """
    data_list: list of input data that is of shape (snapshot, [Torch data object])
    idx: list of indices within data_list that you want to visualize
    """
    for i in idx:
        cell_ids = data_list[i].voronoi_equivalent

        plt.figure()
        plt.imshow(cell_ids % 2)
        plt.savefig(f'training_results_batch3_[norm]/{name}_ssNucleus_idx{i}.png', dpi=200)
        plt.close()

def visualize_snapshot_graph(data_list, idx, name):
    """
    data_list: list of Torch Geometric data objects
    idx: indices to visualize
    """
    for i in idx:
        edges = data_list[i].edge_index
        nodes = data_list[i].cell_pos
 
        nodes_np = nodes.cpu().numpy()
        edges_np = edges.cpu().numpy().T.astype(int)
        
        G = nx.Graph()
        G.add_nodes_from(range(len(nodes_np)))
        G.add_edges_from(edges_np)
        nodes_rot = np.column_stack((nodes_np[:,1], -nodes_np[:,0]))
        pos = dict(enumerate(nodes_rot))
        
        plt.figure()
        nx.draw(G, pos, with_labels=True)

        plt.savefig(f'training_results_batch3_[norm]/{name}_graph_idx{i}.png', dpi=200)
        plt.close()

# see what linear regression on the nucleus area fraction can do
def lin_regression(data_list_test):
    test_y_values = []
    test_phi_values = []
    
    for data in data_list_test:
        test_y_values.append(data.y.item())
        test_phi_values.append(data.area_frac.item())

    # linear fitting
    test_phi_values = np.array(test_phi_values)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(test_phi_values, test_y_values)
    predicted_y_values = slope * test_phi_values + intercept
    correlation = r_value
    mean_squared_error = np.mean((predicted_y_values - test_y_values) ** 2)

    return correlation, mean_squared_error, p_value