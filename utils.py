import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import ndarray as nd
import dgl


def load_data(directory):
    ID = np.loadtxt(directory + '/Similarity_Matrix_DrugsFin.txt')
    IP = np.loadtxt(directory + '/Similarity_Matrix_ProteinsFin.txt')
    return ID, IP


def sample(directory, random_seed):
    all_associations = pd.read_csv(directory + '/data.csv', names=['protein', 'drug', 'label'])
    known_associations = all_associations.loc[all_associations['label'] == 1]
    # print(len(known_associations))
    unknown_associations = all_associations.loc[all_associations['label'] == 0]
    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed, axis=0)

    sample_df = known_associations.append(random_negative)
    sample_df.reset_index(drop=True, inplace=True)

    return sample_df.values


def build_graph(directory, random_seed, ctx):
    # dgl.load_backend('mxnet')
    ID, IP = load_data(directory)
    samples = sample(directory, random_seed)

    print('Building graph ...')
    g = dgl.DGLGraph(multigraph=True)
    g.add_nodes(ID.shape[0] + IP.shape[0])
    node_type = nd.zeros(g.number_of_nodes(), dtype='float32', ctx=ctx)
    node_type[:ID.shape[0]] = 1
    g.ndata['type'] = node_type

    print('Adding drug features ...')
    d_data = nd.zeros(shape=(g.number_of_nodes(), ID.shape[1]), dtype='float32', ctx=ctx)
    d_data[: ID.shape[0], :] = nd.from_numpy(ID)
    g.ndata['d_features'] = d_data

    print('Adding protein features ...')
    p_data = nd.zeros(shape=(g.number_of_nodes(), IP.shape[1]), dtype='float32', ctx=ctx)
    p_data[ID.shape[0]: ID.shape[0]+IP.shape[0], :] = nd.from_numpy(IP)
    g.ndata['p_features'] = p_data

    print('Adding edges ...')
    drug_ids = list(range(1, ID.shape[0] + 1))
    protein_ids = list(range(1, IP.shape[0] + 1))

    drug_ids_invmap = {id_: i for i, id_ in enumerate(drug_ids)}
    protein_ids_invmap = {id_: i for i, id_ in enumerate(protein_ids)}

    sample_drug_vertices = [drug_ids_invmap[id_] for id_ in samples[:, 1]]
    sample_protein_vertices = [protein_ids_invmap[id_] + ID.shape[0] for id_ in samples[:, 0]]
    g.add_edges(sample_drug_vertices, sample_protein_vertices,
                data={'inv': nd.zeros(samples.shape[0], dtype='int32', ctx=ctx),
                      'rating': nd.from_numpy(samples[:, 2].astype('float32')).copyto(ctx)})
    g.add_edges(sample_protein_vertices, sample_drug_vertices,
                data={'inv': nd.zeros(samples.shape[0], dtype='int32', ctx=ctx),
                      'rating': nd.from_numpy(samples[:, 2].astype('float32')).copyto(ctx)})
    g.readonly()
    print('Successfully build graph !!')

    return g, drug_ids_invmap, protein_ids_invmap