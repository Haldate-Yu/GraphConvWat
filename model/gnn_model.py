import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from .layer import GENConvolution, SAGEConv, SSGConv

"""
    Graph convolution based model using multiple GCN layers (with multiple hops).
"""


class m_GCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, latent_dim=96, n_aggr=45, n_hops=1, bias=False, num_layers=2,
                 dropout=0., batch_size=48):
        super(m_GCN, self).__init__()
        self.n_aggr = n_aggr
        self.n_hops = n_hops
        self.batch_size = batch_size
        self.latent = latent_dim
        self.out_dim = out_dim
        self.node_in = torch.nn.Linear(in_dim, latent_dim, bias=bias)
        self.node_out = torch.nn.Linear(latent_dim, out_dim, bias=bias)
        self.edge = torch.nn.Linear(edge_dim, latent_dim, bias=bias)

        self.gcn_aggrs = torch.nn.ModuleList()
        for _ in range(n_aggr):
            gcn = GENConvolution(latent_dim, latent_dim, latent_dim, aggr="add", bias=bias, num_layers=num_layers,
                                 dropout=dropout)
            self.gcn_aggrs.append(gcn)

    def forward(self, data):
        x, y, edge_index, edge_attr = data.x, data.y, data.edge_index, data.edge_attr

        """ Embedding for edge features. """
        edge_attr = self.edge(edge_attr)

        """ Embedding for node features. """
        Z = self.node_in(x)

        # gcn_aggr * multi_hops GCN layers
        """ 
            Mutiple GCN layers.
        """
        for gcn in self.gcn_aggrs:
            """
                Multiple Hops.
            """
            for _ in range(self.n_hops - 1):
                Z = torch.selu(gcn(Z, edge_index, edge_attr, mlp=False))
            Z = torch.selu(gcn(Z, edge_index, edge_attr, mlp=True))

        """ Reconstructing node features through a final dense layer. """
        y_predict = self.node_out(Z)

        del data

        return y_predict

    def cal_loss(self, y, y_predict):
        """ L1 Loss. """
        l1_loss = torch.nn.L1Loss(reduction='mean')
        loss = l1_loss(y, y_predict)
        del y, y_predict
        return loss


class GraphSage(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, dropout=0., use_weight=False):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_weight = use_weight

        self.conv1 = SAGEConv(self.in_dim, self.hid_dim)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(SAGEConv(self.hid_dim, self.hid_dim))
        self.lin1 = torch.nn.Linear(self.hid_dim, self.hid_dim)
        self.lin2 = torch.nn.Linear(self.hid_dim, self.out_dim)

    def reset_parameters(self):
        pass

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.weight
        if self.use_weight:
            x = F.relu(self.conv1(x, edge_index, edge_weight))
        else:
            x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            if self.use_weight:
                x = F.relu(conv(x, edge_index, edge_weight))
            else:
                x = F.relu(conv(x, edge_index))

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x


class SSGC(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, alpha=0.6, dropout=0., use_weight=False):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.alpha = alpha
        self.dropout = dropout
        self.use_weight = use_weight

        self.conv1 = SSGConv(self.in_dim, self.hid_dim, K=self.num_layers, alpha=self.alpha)
        self.lin1 = torch.nn.Linear(self.hid_dim, self.hid_dim)
        self.lin2 = torch.nn.Linear(self.hid_dim, self.out_dim)

    def reset_parameters(self):
        pass

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.weight
        if self.use_weight:
            x = F.relu(self.conv1(x, edge_index, edge_weight))
        else:
            x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x)

        return x
