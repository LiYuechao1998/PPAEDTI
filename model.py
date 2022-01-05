import math

import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn
import dgl
import numpy as np
from dgl.nn.mxnet import APPNPConv, GATConv, TAGConv,SAGEConv,GraphConv


class GNNMDA(nn.Block):
    def __init__(self, encoder, decoder):
        super(GNNMDA, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, G, drug, protein):
        h = self.encoder(G)
        # print('h=' + len(h).__str__())
        h_drug = h[drug]
        # print('h_drug=' + len(h_drug).__str__())
        h_protein = h[protein]
        # print('h_protein=' + len(h_protein).__str__())
        return self.decoder(h_drug, h_protein)


class GraphEncoder(nn.Block):
    def __init__(self, embedding_size, n_layers, G, aggregator, dropout, slope, ctx):
        super(GraphEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.n = n_layers
        self.G = G
        self.aggregator = aggregator
        self.dropout = dropout
        self.ctx = ctx
        self.drug_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1).astype(np.int64).copyto(ctx)
        self.protein_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0).astype(np.int64).copyto(ctx)
        self.drug_emb = DrugEmbedding(embedding_size, dropout)
        self.protein_emb = ProteinEmbedding(embedding_size, dropout)
        if aggregator == 'APPNPConv':
            self.appnpconvlayers = nn.Sequential()
            self.appnpconvlayers.add(APPNPConv(k=self.n, alpha=0.6, edge_drop=dropout))
            # self.W = nn.Dense(embedding_size)
            self.leakyrelu = nn.LeakyReLU(slope)
            # self.dropout1 = nn.Dropout(dropout)
        elif aggregator == 'SAGEConv':
            if self.n_layers == 1:
                self.sageconvlayers = nn.Sequential()
                self.sageconvlayers.add(SAGEConv(in_feats=self.embedding_size, out_feats=self.embedding_size,
                                                 aggregator_type='mean', feat_drop=self.dropout, bias=True,
                                                 norm=None, activation=nn.LeakyReLU(slope)))
            elif self.n_layers == 2:
                self.sageconvlayers = nn.Sequential()
                self.sageconvlayers.add(SAGEConv(in_feats=self.embedding_size, out_feats=self.embedding_size,
                                                 aggregator_type='mean',feat_drop=self.dropout, bias=True,
                                                 norm=None,activation=nn.LeakyReLU(slope)))
                self.sageconvlayers1 = nn.Sequential()
                self.sageconvlayers1.add(SAGEConv(in_feats=self.embedding_size, out_feats=self.embedding_size,
                                                 aggregator_type='mean', feat_drop=self.dropout, bias=True,
                                                 norm=None, activation=nn.LeakyReLU(slope)))
            else:
                raise NotImplementedError
        elif aggregator == 'GraphConv':
            self.dropout1 = nn.Dropout(dropout)
            if self.n_layers == 1:
                self.graphconvlayers = nn.Sequential()
                self.graphconvlayers.add(GraphConv(in_feats=self.embedding_size, out_feats=self.embedding_size,
                                                   norm='both', weight=True, bias=True, activation=nn.LeakyReLU(slope)))
            elif self.n_layers == 2:
                self.graphconvlayers = nn.Sequential()
                self.graphconvlayers.add(GraphConv(in_feats=self.embedding_size, out_feats=self.embedding_size,
                                                   norm='both', weight=True, bias=True, activation=nn.LeakyReLU(slope)))
                self.graphconvlayers1 = nn.Sequential()
                self.graphconvlayers1.add(GraphConv(in_feats=self.embedding_size, out_feats=self.embedding_size,
                                                   norm='both', weight=True, bias=True, activation=nn.LeakyReLU(slope)))
            else:
                raise NotImplementedError
        elif aggregator == 'TAGConv':
            self.tagconvlayers = nn.Sequential()
            self.tagconvlayers.add(TAGConv(in_feats=self.embedding_size, out_feats=self.embedding_size, k=self.n, bias=True, activation=nn.LeakyReLU(slope)))
            self.dropout1 = nn.Dropout(dropout)
        elif aggregator == 'GATConv':
            if self.n_layers == 1:
                self.gatconvlayers = nn.Sequential()
                self.gatconvlayers.add(GATConv(in_feats=self.embedding_size, out_feats=self.embedding_size, num_heads=1,
                                                feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=False,
                                                activation=nn.LeakyReLU(slope)))
            elif self.n_layers == 2:
                self.gatconvlayers = nn.Sequential()
                self.gatconvlayers.add(GATConv(in_feats=self.embedding_size, out_feats=self.embedding_size, num_heads=1,
                                               feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=False,
                                               activation=nn.LeakyReLU(slope)))
                self.gatconvlayers1 = nn.Sequential()
                self.gatconvlayers1.add(GATConv(in_feats=self.embedding_size, out_feats=self.embedding_size, num_heads=1,
                                               feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=False,
                                               activation=nn.LeakyReLU(slope)))
            else:
                raise NotImplementedError
        else:
            pass
            # raise NotImplementedError

    def forward(self, G):
        # Generate embedding on disease nodes and mirna nodes
        assert G.number_of_nodes() == self.G.number_of_nodes()
        G.apply_nodes(lambda nodes: {'h': self.drug_emb(nodes.data)}, self.drug_nodes)
        G.apply_nodes(lambda nodes: {'h': self.protein_emb(nodes.data)}, self.protein_nodes)
        if self.aggregator == 'APPNPConv':
            feat = G.ndata['h']
            for layer in self.appnpconvlayers:
                feat = layer(G, feat)
            # feat = self.dropout1(self.leakyrelu(self.W(feat)))
            feat = self.leakyrelu(feat)
            G.ndata['h'] = feat
        elif self.aggregator == 'SAGEConv':
            if self.n_layers == 1:
                for layer in self.sageconvlayers:
                    feat = G.ndata['h']
                    feat = layer(G, feat)
                    G.ndata['h'] = feat
            elif self.n_layers == 2:
                for layer in self.sageconvlayers:
                    feat = G.ndata['h']
                    feat = layer(G,feat)
                    G.ndata['h'] = feat
                for layer in self.sageconvlayers1:
                    feat = G.ndata['h']
                    feat = layer(G,feat)
                    G.ndata['h'] = feat
        elif self.aggregator == 'GraphConv':
            if self.n_layers == 1:
                for layer in self.graphconvlayers:
                    feat = G.ndata['h']
                    feat = layer(G, feat)
                    feat = self.dropout1(feat)
                    G.ndata['h'] = feat
            elif self.n_layers == 2:
                for layer in self.graphconvlayers:
                    feat = G.ndata['h']
                    feat = layer(G,feat)
                    feat = self.dropout1(feat)
                    G.ndata['h'] = feat
                for layer in self.graphconvlayers1:
                    feat = G.ndata['h']
                    feat = layer(G,feat)
                    feat = self.dropout1(feat)
                    G.ndata['h'] = feat
        elif self.aggregator == 'TAGConv':
            feat = G.ndata['h']
            for layer in self.tagconvlayers:
                feat = layer(G,feat)
            feat = self.dropout1(feat)
            G.ndata['h'] = feat
        elif self.aggregator == 'GATConv':
            if self.n_layers == 1:
                for layer in self.gatconvlayers:
                    feat = G.ndata['h']
                    feat = layer(G, feat)
                    feat = feat.reshape((G.number_of_nodes(), self.embedding_size))
                    G.ndata['h'] = feat
            elif self.n_layers == 2:
                for layer in self.gatconvlayers:
                    feat = G.ndata['h']
                    feat = layer(G, feat)
                    feat = feat.reshape((G.number_of_nodes(), self.embedding_size))
                    G.ndata['h'] = feat
                for layer in self.gatconvlayers1:
                    feat = G.ndata['h']
                    feat = layer(G, feat)
                    feat = feat.reshape((G.number_of_nodes(), self.embedding_size))
                    G.ndata['h'] = feat
        else:
            pass
            # raise NotImplementedError
        return G.ndata['h']


class DrugEmbedding(nn.Block):
    def __init__(self, embedding_size, dropout):
        super(DrugEmbedding, self).__init__()

        seq = nn.Sequential()
        with seq.name_scope():
            seq.add(nn.Dense(embedding_size, use_bias=False))
            seq.add(nn.Dropout(dropout))
        self.proj_drug = seq

    def forward(self, ndata):
        extra_repr = self.proj_drug(ndata['d_features'])
        return extra_repr


class ProteinEmbedding(nn.Block):
    def __init__(self, embedding_size, dropout):
        super(ProteinEmbedding, self).__init__()

        seq = nn.Sequential()
        with seq.name_scope():
            seq.add(nn.Dense(embedding_size, use_bias=False))
            seq.add(nn.Dropout(dropout))
        self.proj_protein = seq

    def forward(self, ndata):
        extra_repr = self.proj_protein(ndata['p_features'])
        return extra_repr


class BilinearDecoder(nn.Block):
    def __init__(self, feature_size):
        super(BilinearDecoder, self).__init__()
        self.activation = nn.Activation('sigmoid')
        with self.name_scope():
            self.W = self.params.get('dot_weights', shape=(feature_size, feature_size))

    def forward(self, h_drug, h_protein):
        results_mask = self.activation((nd.dot(h_drug, self.W.data()) * h_protein).sum(1))
        return results_mask

# class BilinearDecoder(nn.Block):
#     def __init__(self, feature_size):
#         super(BilinearDecoder, self).__init__()
#         self.activation = nn.Activation('sigmoid')
#
#     def forward(self, h_drug, h_protein):
#         results_mask = self.activation((h_drug* h_protein).sum(1))
#         return results_mask