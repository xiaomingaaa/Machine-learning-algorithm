'''
Author: your name
Date: 2021-04-16 22:14:00
LastEditTime: 2021-04-18 11:33:22
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /ml/GNN/gcn.py
'''
import torch.nn as nn
import torch.nn.functional as F
from dgl.function as fn
import dgl
from dgl.utils import expand_as_pair
from dgl.utils import check_eq_shape


class GraphSage(nn.Module):
    def __init__(self, in_feats, out_feats, aggregator_type, bias=True, norm=None, activation=None):
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.activation = activation

        if self._aggre_type == 'max_pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if self._aggre_type == 'lstm':
            self.lstm = nn.LSTM(self._in_src_feats,
                                self._in_src_feats, batch_first=True)
        if self._aggre_type in ['mean', 'max_pool', 'lstm']:
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'max_pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, graph: dgl.graph, feat):
        with graph.local_scope():
            # 确定源节点和目标节点
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._aggre_type == 'mean':
                graph.srcdata['h'] = feat_src
                graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']
            elif self._aggre_type == 'gcn':
                check_eq_shape(feat)
                graph.srcdata['h'] = feat_src
                graph.dstdata['h'] = feat_dst
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'neigh'))
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata['neigh'] +
                           graph.dstdata['h'])/(degs.unsqueeze(-1)+1)
            elif self._aggre_type == 'max_pool':
                graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
                graph.update_all(fn.copy_u('h', 'm'), fn.max('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']
            h_self = graph.srcdata['h']
            rst = self.fc_self(h_self)+self.fc_neigh(h_neigh)
            if self.norm is not None:
                rst = self.norm(rst)
            return rst
