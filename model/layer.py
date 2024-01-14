from typing import List, Optional, Union, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (Dropout, Sequential, SELU)
from torch_geometric.utils import spmm
from torch_scatter import scatter
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size


class MLP(Sequential):
    def __init__(self, dims: List[int], bias: bool = True, dropout: float = 0., activ=SELU()):
        m = []
        for i in range(1, len(dims)):
            m.append(Linear(dims[i - 1], dims[i], bias=bias))

            if i < len(dims) - 1:
                m.append(activ)
                m.append(Dropout(dropout))

        super().__init__(*m)


class GENConvolution(MessagePassing):
    r"""
    Args:
        in_dim (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_dim (int): Size of each output sample.
        edge_dim (int): Size of edge features.
        aggr (str, optional): The aggregation scheme to use (:obj:`"softmax"`,
            :obj:`"softmax_sg"`, :obj:`"power"`, :obj:`"add"`, :obj:`"mean"`,
            :obj:`max`). (default: :obj:`"softmax"`)
        num_layers (int, optional): The number of MLP layers.
            (default: :obj:`2`)
        eps (float, optional): The epsilon value of the message construction
            function. (default: :obj:`1e-7`)
        bias (bool, optional): If set to :obj:`False`, will not use bias.
            (default: :obj:`True`)
        dropout (float, optional): Percentage of neurons to be dropped in MLP.
            (default: :obj:`0.`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GenMessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{in}), (|\mathcal{V_t}|, F_{t})`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge attributes :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int,
                 aggr: str = 'add', num_layers: int = 2, eps: float = 1e-7,
                 bias: bool = True, dropout: float = 0., **kwargs):

        kwargs.setdefault('aggr', None)
        super().__init__(**kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.aggr = aggr
        self.eps = eps

        assert aggr in ['add', 'mean', 'max']

        dims = [self.in_dim]
        for i in range(num_layers - 1):
            dims.append(2 * in_dim)
        dims.append(self.out_dim)
        self.mlp = MLP(dims, bias=bias, dropout=dropout)

        """ Added a linear layer to manage dimensionality """
        self.res = Linear(in_dim + edge_dim, in_dim, bias=bias)

    def reset_parameters(self):
        if self.msg_norm is not None:
            self.msg_norm.reset_parameters()
        if self.t and isinstance(self.t, Tensor):
            self.t.data.fill_(self.initial_t)
        if self.p and isinstance(self.p, Tensor):
            self.p.data.fill_(self.initial_p)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                residual: bool = True, mlp: bool = True) -> Tensor:
        """"""

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        x_in = x[0]

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        sndr_node_attr = torch.gather(x_in, 0, edge_index[0:1, :].repeat(x_in.shape[1], 1).T)
        rcvr_node_attr = torch.gather(x_in, 0, edge_index[1:2, :].repeat(x_in.shape[1], 1).T)
        edge_attr = edge_attr + (sndr_node_attr - rcvr_node_attr).abs()

        latent = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        """ Added a linear layer to manage dimensionality """
        if mlp:
            latent = self.res(latent)
        else:
            latent = torch.tanh(self.res(latent))

        del sndr_node_attr, rcvr_node_attr

        if residual:
            latent = latent + x[1]

        del x, edge_index, edge_attr
        if mlp:
            latent = self.mlp(latent)
        return latent

    def message(self, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        """ Concatenating edge features instead of adding those to node features """
        msg = x_j if edge_attr is None else torch.cat((x_j, edge_attr), dim=1)
        del x_j, edge_attr
        return F.selu(msg) + self.eps

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:

        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                       reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_dim}, '
                f'{self.out_dim}, aggr={self.aggr})')


class SAGEConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

            # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)


class SSGConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, alpha: float,
                 K: int = 1, cached: bool = False, add_self_loops: bool = True,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.K = K
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._cached_h = None

        self.lin = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        self._cached_h = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        cache = self._cached_h
        if cache is None:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, self.flow, dtype=x.dtype)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, self.flow, dtype=x.dtype)

            h = x * self.alpha
            for k in range(self.K):
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
                h = h + (1 - self.alpha) / self.K * x
            if self.cached:
                self._cached_h = h
        else:
            h = cache.detach()

        return self.lin(h)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K}, alpha={self.alpha})')
