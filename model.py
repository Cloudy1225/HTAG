import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import apply_each
from dgl.nn.pytorch import GATConv, GraphConv, SAGEConv, HeteroGraphConv, HeteroLinear


class MLP(nn.Module):
    def __init__(
            self,
            in_dim,
            hid_dims,
            out_dim,
            dropout,
            activation,
            norm_type='none',
    ):
        super(MLP, self).__init__()
        self.norm_type = norm_type

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

        hid_dims = [in_dim] + hid_dims + [out_dim]
        for i in range(len(hid_dims) - 1):
            self.layers.append(nn.Linear(hid_dims[i], hid_dims[i + 1]))

        for i in range(len(hid_dims) - 2):
            if self.norm_type == 'batch':
                self.norms.append(nn.BatchNorm1d(hid_dims[i + 1]))
            elif self.norm_type == 'layer':
                self.norms.append(nn.LayerNorm(hid_dims[i + 1]))

    def forward(self, feats):
        h = feats
        for l, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(h)
            if l != len(self.layers) - 1:
                if self.norm_type != 'none':
                    h = self.norms[l](h)
                h = self.activation(h)
        return h


class GCN(nn.Module):
    def __init__(
            self,
            in_dim,
            hid_dims,
            out_dim,
            dropout,
            activation,
            norm_type='none',
    ):
        super(GCN, self).__init__()
        self.norm_type = norm_type

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        self.activation = getattr(nn, activation)()

        hid_dims = [in_dim] + hid_dims
        for i in range(len(hid_dims) - 1):
            self.layers.append(GraphConv(hid_dims[i], hid_dims[i + 1]))

        for i in range(len(hid_dims) - 2):
            if self.norm_type == 'batch':
                self.norms.append(nn.BatchNorm1d(hid_dims[i + 1]))
            elif self.norm_type == 'layer':
                self.norms.append(nn.LayerNorm(hid_dims[i + 1]))

        self.classifier = nn.Linear(hid_dims[-1], out_dim)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = self.dropout(h)
            h = layer(block, h)
            if l != len(self.layers) - 1:
                if self.norm_type != 'none':
                    h = self.norms[l](h)
                h = self.activation(h)

        logits = self.classifier(h)

        return logits


class SAGE(nn.Module):
    def __init__(
            self,
            in_dim,
            hid_dims,
            out_dim,
            dropout,
            activation,
            norm_type='none',
    ):
        super(SAGE, self).__init__()
        self.norm_type = norm_type

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        self.activation = getattr(nn, activation)()

        hid_dims = [in_dim] + hid_dims
        for i in range(len(hid_dims) - 1):
            self.layers.append(SAGEConv(hid_dims[i], hid_dims[i + 1], aggregator_type='gcn'))

        for i in range(len(hid_dims) - 2):
            if self.norm_type == 'batch':
                self.norms.append(nn.BatchNorm1d(hid_dims[i + 1]))
            elif self.norm_type == 'layer':
                self.norms.append(nn.LayerNorm(hid_dims[i + 1]))

        self.classifier = nn.Linear(hid_dims[-1], out_dim)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = self.dropout(h)
            h = layer(block, h)
            if l != len(self.layers) - 1:
                if self.norm_type != 'none':
                    h = self.norms[l](h)
                h = self.activation(h)

        logits = self.classifier(h)

        return logits


class GAT(nn.Module):
    def __init__(
            self,
            in_dim,
            hid_dims,
            out_dim,
            num_heads,
            dropout,
            attn_drop,
            activation,
            norm_type='none',
    ):
        super(GAT, self).__init__()
        self.norm_type = norm_type

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        self.activation = getattr(nn, activation)()

        hid_dims = [in_dim] + hid_dims
        for i in range(len(hid_dims) - 1):
            self.layers.append(GATConv(hid_dims[i], hid_dims[i + 1] // num_heads,
                                       num_heads, attn_drop=attn_drop))

        for i in range(len(hid_dims) - 2):
            if self.norm_type == 'batch':
                self.norms.append(nn.BatchNorm1d(hid_dims[i + 1]))
            elif self.norm_type == 'layer':
                self.norms.append(nn.LayerNorm(hid_dims[i + 1]))

        self.classifier = nn.Linear(hid_dims[-1], out_dim)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = self.dropout(h)
            h = layer(block, h)
            h = h.flatten(1)
            if l != len(self.layers) - 1:
                if self.norm_type != 'none':
                    h = self.norms[l](h)
                h = self.activation(h)

        logits = self.classifier(h)

        return logits


class RGCN(nn.Module):
    def __init__(self, etypes, ntypes, target,
                 in_dim,
                 hid_dims,
                 out_dim,
                 dropout,
                 activation,
                 norm_type='none',
                 skip_connection=False, ):
        super(RGCN, self).__init__()
        self.target = target
        self.norm_type = norm_type
        self.skip_connection = skip_connection

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        if self.skip_connection:
            self.skips = nn.ModuleList()

        self.activation = getattr(nn, activation)()

        hid_dims = [in_dim] + hid_dims
        for i in range(len(hid_dims) - 1):
            self.layers.append(HeteroGraphConv(
                {etype: GraphConv(hid_dims[i], hid_dims[i + 1], norm='right')
                 for etype in etypes}, aggregate='mean'))
            if self.skip_connection:
                self.skips.append(nn.Linear(hid_dims[i], hid_dims[i + 1]))

        for i in range(len(hid_dims) - 2):
            if self.norm_type == 'batch':
                self.norms.append(nn.ModuleDict({
                    ntype: nn.BatchNorm1d(hid_dims[i + 1]) for ntype in ntypes
                }))
            elif self.norm_type == 'layer':
                self.norms.append(nn.ModuleDict({
                    ntype: nn.LayerNorm(hid_dims[i + 1]) for ntype in ntypes
                }))

        self.classifier = nn.Linear(hid_dims[-1], out_dim)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = apply_each(h, self.dropout)
            if self.skip_connection:
                num_dst_nodes = block.num_dst_nodes(self.target)
                h_skip = self.skips[l](h[self.target][:num_dst_nodes])
            h = layer(block, h)
            h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1]))
            if self.skip_connection:
                h[self.target] = h[self.target] + h_skip
            if l != len(self.layers) - 1:
                if self.norm_type != 'none':
                    h = {ntype: norm(h[ntype]) for ntype, norm in self.norms[l].items()}
                h = apply_each(h, self.activation)

        logits = self.classifier(h[self.target])

        return logits


class RSAGE(nn.Module):
    def __init__(self, etypes, ntypes, target,
                 in_dim,
                 hid_dims,
                 out_dim,
                 dropout,
                 activation,
                 norm_type='none',
                 skip_connection=False, ):
        super(RSAGE, self).__init__()
        self.target = target
        self.norm_type = norm_type
        self.skip_connection = skip_connection

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        if self.skip_connection:
            self.skips = nn.ModuleList()

        self.activation = getattr(nn, activation)()

        hid_dims = [in_dim] + hid_dims
        for i in range(len(hid_dims) - 1):
            self.layers.append(HeteroGraphConv(
                {etype: SAGEConv(hid_dims[i], hid_dims[i + 1], aggregator_type='gcn')
                 for etype in etypes}, aggregate='mean'))
            if self.skip_connection:
                self.skips.append(nn.Linear(hid_dims[i], hid_dims[i + 1]))

        for i in range(len(hid_dims) - 2):
            if self.norm_type == 'batch':
                self.norms.append(nn.ModuleDict({
                    ntype: nn.BatchNorm1d(hid_dims[i + 1]) for ntype in ntypes
                }))

            elif self.norm_type == 'layer':
                self.norms.append(nn.ModuleDict({
                    ntype: nn.LayerNorm(hid_dims[i + 1]) for ntype in ntypes
                }))

        self.classifier = nn.Linear(hid_dims[-1], out_dim)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = apply_each(h, self.dropout)
            if self.skip_connection:
                num_dst_nodes = block.num_dst_nodes(self.target)
                h_skip = self.skips[l](h[self.target][:num_dst_nodes])
            h = layer(block, h)
            h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1]))
            if self.skip_connection:
                h[self.target] = h[self.target] + h_skip
            if l != len(self.layers) - 1:
                if self.norm_type != 'none':
                    h = {ntype: norm(h[ntype]) for ntype, norm in self.norms[l].items()}
                h = apply_each(h, self.activation)

        logits = self.classifier(h[self.target])

        return logits


class RGAT(nn.Module):
    def __init__(self, etypes, ntypes, target,
                 in_dim,
                 hid_dims,
                 out_dim,
                 num_heads,
                 dropout,
                 attn_drop,
                 activation,
                 norm_type='none',
                 skip_connection=False, ):
        super(RGAT, self).__init__()

        self.target = target
        self.norm_type = norm_type
        self.skip_connection = skip_connection

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        if self.skip_connection:
            self.skips = nn.ModuleList()

        self.activation = getattr(nn, activation)()

        hid_dims = [in_dim] + hid_dims
        for i in range(len(hid_dims) - 1):
            self.layers.append(HeteroGraphConv(
                {etype: GATConv(hid_dims[i], hid_dims[i + 1] // num_heads, num_heads, attn_drop=attn_drop)
                 for etype in etypes}, aggregate='mean'))
            if self.skip_connection:
                self.skips.append(nn.Linear(hid_dims[i], hid_dims[i + 1]))

        for i in range(len(hid_dims) - 2):
            if self.norm_type == 'batch':
                self.norms.append(nn.ModuleDict({
                    ntype: nn.BatchNorm1d(hid_dims[i + 1]) for ntype in ntypes
                }))
            elif self.norm_type == 'layer':
                self.norms.append(nn.ModuleDict({
                    ntype: nn.LayerNorm(hid_dims[i + 1]) for ntype in ntypes
                }))

        self.classifier = nn.Linear(hid_dims[-1], out_dim)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = apply_each(h, self.dropout)
            if self.skip_connection:
                num_dst_nodes = block.num_dst_nodes(self.target)
                h_skip = self.skips[l](h[self.target][:num_dst_nodes])
            h = layer(block, h)
            h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))
            if self.skip_connection:
                h[self.target] = h[self.target] + h_skip
            if l != len(self.layers) - 1:
                if self.norm_type != 'none':
                    h = {ntype: norm(h[ntype]) for ntype, norm in self.norms[l].items()}
                h = apply_each(h, self.activation)

        logits = self.classifier(h[self.target])

        return logits


class ieHGCN(nn.Module):
    def __init__(self, etypes, ntypes, target,
                 in_dim,
                 hid_dims,
                 out_dim,
                 dropout,
                 activation,
                 norm_type='none',
                 skip_connection=False, ):
        super(ieHGCN, self).__init__()
        self.target = target
        self.norm_type = norm_type
        self.skip_connection = skip_connection

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        if self.skip_connection:
            self.skips = nn.ModuleList()

        self.activation = getattr(nn, activation)()

        hid_dims = [in_dim] + hid_dims
        for i in range(len(hid_dims) - 1):
            self.layers.append(ieHGCNConv(hid_dims[i], hid_dims[i + 1], hid_dims[i + 1], ntypes, etypes))
            if self.skip_connection:
                self.skips.append(nn.Linear(hid_dims[i], hid_dims[i + 1]))

        for i in range(len(hid_dims) - 2):
            if self.norm_type == 'batch':
                self.norms.append(nn.ModuleDict({
                    ntype: nn.BatchNorm1d(hid_dims[i + 1]) for ntype in ntypes
                }))
            elif self.norm_type == 'layer':
                self.norms.append(nn.ModuleDict({
                    ntype: nn.LayerNorm(hid_dims[i + 1]) for ntype in ntypes
                }))

        self.classifier = nn.Linear(hid_dims[-1], out_dim)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = apply_each(h, self.dropout)
            if self.skip_connection:
                num_dst_nodes = block.num_dst_nodes(self.target)
                h_skip = self.skips[l](h[self.target][:num_dst_nodes])
            h = layer(block, h)
            h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1]))
            if self.skip_connection:
                h[self.target] = h[self.target] + h_skip
            if l != len(self.layers) - 1:
                if self.norm_type != 'none':
                    h = {ntype: norm(h[ntype]) for ntype, norm in self.norms[l].items()}
                h = apply_each(h, self.activation)

        logits = self.classifier(h[self.target])

        return logits


# This class is adapted from https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/models/ieHGCN.py#L152.
class ieHGCNConv(nn.Module):
    r"""
    The ieHGCN convolution layer.

    Parameters
    ----------
    in_size: int
        the input dimension
    out_size: int
        the output dimension
    attn_size: int
        the dimension of attention vector
    ntypes: list
        the node type list of a heterogeneous graph
    etypes: list
        the edge type list of a heterogeneous graph
    bias: boolean
        whether we need bias vector
    """

    def __init__(self, in_size, out_size, attn_size, ntypes, etypes, bias=False):
        super(ieHGCNConv, self).__init__()
        self.bias = bias
        node_size = {}
        for ntype in ntypes:
            node_size[ntype] = in_size
        attn_vector = {}
        for ntype in ntypes:
            attn_vector[ntype] = attn_size
        self.W_self = HeteroLinear(node_size, out_size)
        self.W_al = HeteroLinear(attn_vector, 1)
        self.W_ar = HeteroLinear(attn_vector, 1)

        self.in_size = in_size
        self.out_size = out_size
        self.attn_size = attn_size
        mods = {
            etype: GraphConv(in_size, out_size, norm='right',
                             weight=True, bias=True, allow_zero_in_degree=True)
            for etype in etypes
        }
        self.mods = nn.ModuleDict(mods)

        self.linear_q = nn.ModuleDict({ntype: nn.Linear(out_size, attn_size) for ntype in ntypes})
        self.linear_k = nn.ModuleDict({ntype: nn.Linear(out_size, attn_size) for ntype in ntypes})

        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_size))
            nn.init.zeros_(self.h_bias)

    def forward(self, hg, h_dict):
        """
        The forward part of the ieHGCNConv.

        Parameters
        ----------
        hg : object or list[block]
            the dgl heterogeneous graph or the list of blocks
        h_dict: dict
            the feature dict of different node types

        Returns
        -------
        dict
            The embeddings after final aggregation.
        """
        outputs = {ntype: [] for ntype in hg.dsttypes}
        if hg.is_block:
            src_inputs = h_dict
            dst_inputs = {k: v[:hg.number_of_dst_nodes(k)] for k, v in h_dict.items()}
        else:
            src_inputs = h_dict
            dst_inputs = h_dict
        with hg.local_scope():
            hg.ndata['h'] = h_dict
            # formulas (2)-1
            dst_inputs = self.W_self(dst_inputs)
            query = {}
            key = {}
            attn = {}
            attention = {}

            # formulas (3)-1 and (3)-2
            for ntype in hg.dsttypes:
                query[ntype] = self.linear_q[ntype](dst_inputs[ntype])
                key[ntype] = self.linear_k[ntype](dst_inputs[ntype])
            # formulas (4)-1
            h_l = self.W_al(key)
            h_r = self.W_ar(query)
            for ntype in hg.dsttypes:
                attention[ntype] = F.elu(h_l[ntype] + h_r[ntype])
                attention[ntype] = attention[ntype].unsqueeze(0)

            for srctype, etype, dsttype in hg.canonical_etypes:
                rel_graph = hg[srctype, etype, dsttype]
                if srctype not in h_dict:
                    continue
                # formulas (2)-2
                dstdata = self.mods[etype](
                    rel_graph,
                    (src_inputs[srctype], dst_inputs[dsttype])
                )
                outputs[dsttype].append(dstdata)
                # formulas (3)-3
                attn[dsttype] = self.linear_k[dsttype](dstdata)
                # formulas (4)-2
                h_attn = self.W_al(attn)
                attn.clear()
                edge_attention = F.elu(h_attn[dsttype] + h_r[dsttype])
                attention[dsttype] = torch.cat((attention[dsttype], edge_attention.unsqueeze(0)))

            # formulas (5)
            for ntype in hg.dsttypes:
                attention[ntype] = F.softmax(attention[ntype], dim=0)

            # formulas (6)
            rst = {ntype: 0 for ntype in hg.dsttypes}
            for ntype, data in outputs.items():
                data = [dst_inputs[ntype]] + data
                if len(data) != 0:
                    for i in range(len(data)):
                        aggregation = torch.mul(data[i], attention[ntype][i])
                        rst[ntype] = aggregation + rst[ntype]

        def _apply(h):
            if self.bias:
                h = h + self.h_bias
            return h

        return apply_each(rst, _apply)
