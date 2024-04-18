import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (register_node_encoder,
                                               register_edge_encoder)

"""
=== Description of the PBFSSuperpixels dataset === 
Each graph is a tuple (x, edge_attr, edge_index, y)
Shape of x : [num_nodes, 120]
Shape of edge_attr : [num_edges, *] or [num_edges, *]
Shape of edge_index : [2, num_edges]
Shape of y : [num_nodes]
"""

PBFS_node_input_dim = 139
# PBFS_edge_input_dim = 1 or 2; defined in class PBFSEdgeEncoder

@register_node_encoder('PBFSNode')
class PBFSNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.encoder = torch.nn.Linear(PBFS_node_input_dim, emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        batch.x = self.encoder(batch.x)

        return batch


@register_edge_encoder('PBFSEdge')
class PBFSEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        PBFS_edge_input_dim = 18 if cfg.dataset.name == 'edge_wt_region_boundary' else 1
        self.encoder = torch.nn.Linear(PBFS_edge_input_dim, emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        batch.edge_attr = self.encoder(batch.edge_attr)
        return batch
