# The GNN model that will be trained (copied from Yang et al.)

import torch
from torch.nn import ModuleList, Linear
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import PNAConv, BatchNorm, global_mean_pool, PairNorm
from torch_geometric.utils import degree


# Compute degree histogram over training data

def pre_deg(pre_data):

    all_degrees = []

    for data in pre_data:
        row, col = data.edge_index

        # Sum in both directions since edges are undirectional
        d = degree(row, num_nodes=data.num_nodes, dtype=torch.long) + \
            degree(col, num_nodes=data.num_nodes, dtype=torch.long)

        all_degrees.append(d)

    all_degrees = torch.cat(all_degrees)
    return torch.bincount(all_degrees)
    

# Class that defines the GNN

class myPNA(torch.nn.Module):
    def __init__(self, data_list, in_channels, hidden_channels, num_layers):
        super(myPNA, self).__init__()

        # initialize parameters for PNA
        deg=pre_deg(pre_data=data_list)
        print(deg)
        aggregators = ['sum','mean','std','max','min', 'var']
        scalers = ['identity', 'amplification', 'attenuation', 'linear', 'inverse_linear']

        # initialize module list
        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        # first layer
        conv = PNAConv(in_channels=in_channels, out_channels=hidden_channels, edge_dim=1,
                       aggregators=aggregators, scalers=scalers, deg=deg,
                       towers=3, pre_layers=1, post_layers=1,
                       divide_input=False)  
        self.convs.append(conv)
        self.batch_norms.append(BatchNorm(hidden_channels))

        # hidden layers
        for _ in range(num_layers-1):
            conv = PNAConv(in_channels=hidden_channels, out_channels=hidden_channels, edge_dim=1,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           towers=3, pre_layers=1, post_layers=1,
                           divide_input=False)  
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))
        
        # last linear layer
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_attr = edge_attr.view(-1, 1)
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr)
            x = batch_norm(x)
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return F.softplus(x)