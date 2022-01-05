import time
import random
import numpy as np
import pandas as pd
import math
import mxnet as mx
from mxnet import ndarray as nd, gluon, autograd
from mxnet.gluon import loss as gloss
import dgl
from sklearn.model_selection import KFold
from sklearn import metrics
from utils import build_graph, sample, load_data
from model import GNNMDA, GraphEncoder, BilinearDecoder

def Train(directory, epochs, aggregator, embedding_size, layers, dropout, slope, lr, wd, random_seed, ctx):
    dgl.load_backend('mxnet')
    random.seed(random_seed)
    np.random.seed(random_seed)
    mx.random.seed(random_seed)
    g, drug_ids_invmap, protein_ids_invmap = build_graph(directory, random_seed=random_seed, ctx=ctx)
    samples = sample(directory, random_seed=random_seed)
    ID, IP = load_data(directory)

    print('## vertices:', g.number_of_nodes())
    print('## edges:', g.number_of_edges())
    print('## Drug nodes:', nd.sum(g.ndata['type'] == 1).asnumpy())
    print('## Protein nodes:', nd.sum(g.ndata['type'] == 0).asnumpy())

    samples_df = pd.DataFrame(samples, columns=['protein', 'drug', 'label'])
    sample_drug_vertices = [drug_ids_invmap[id_] for id_ in samples[:, 1]]
    sample_protein_vertices = [protein_ids_invmap[id_] + ID.shape[0] for id_ in samples[:, 0]]

    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    train_index = []
    test_index = []
    for train_idx, test_idx in kf.split(samples[:, 2]):
        train_index.append(train_idx)
        test_index.append(test_idx)
    auc_result = []
    acc_result = []
    pre_result = []
    recall_result = []
    f1_result = []
    fprs = []
    tprs = []
    for i in range(len(train_index)):
        print('------------------------------------------------------------------------------------------------------')
        print('Training for Fold ', i + 1)

        samples_df['train'] = 0
        samples_df['test'] = 0
        samples_df['train'].iloc[train_index[i]] = 1
        samples_df['test'].iloc[test_index[i]] = 1

        train_tensor = nd.from_numpy(samples_df['train'].values.astype('int32')).copyto(ctx)
        test_tensor = nd.from_numpy(samples_df['test'].values.astype('int32')).copyto(ctx)

        edge_data = {'train': train_tensor,
                     'test': test_tensor}

        g.edges[sample_drug_vertices, sample_protein_vertices].data.update(edge_data)
        g.edges[sample_protein_vertices, sample_drug_vertices].data.update(edge_data)
        train_eid = g.filter_edges(lambda edges: edges.data['train']).astype('int64')
        g_train = g.edge_subgraph(train_eid, preserve_nodes=True)
        g_train.copy_from_parent()
        # get the training set
        rating_train = g_train.edata['rating']
        src_train, dst_train = g_train.all_edges()
        # print(src_train)
        # print(dst_train)
        # get the testing edge set
        test_eid = g.filter_edges(lambda edges: edges.data['test']).astype('int64')
        src_test, dst_test = g.find_edges(test_eid)
        # print(src_test)
        # print(dst_test)
        rating_test = g.edges[test_eid].data['rating']
        src_train = src_train.copyto(ctx)
        src_test = src_test.copyto(ctx)
        dst_train = dst_train.copyto(ctx)
        dst_test = dst_test.copyto(ctx)
        print('## Training edges:', len(train_eid))
        print('## Testing edges:', len(test_eid))

        # Train the model
        model = GNNMDA(GraphEncoder(embedding_size=embedding_size, n_layers=layers, G=g_train, aggregator=aggregator,
                                    dropout=dropout, slope=slope, ctx=ctx),
                       BilinearDecoder(feature_size=embedding_size))

        print("model successful")
        # print(model)
        # model.collect_params().initialize(init=mx.init.Xavier(magnitude=math.sqrt(2.0)), ctx=ctx)
        model.initialize(init=mx.init.Xavier(magnitude=math.sqrt(2.0)), ctx=ctx)
        cross_entropy = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
        trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd})
        # print(model.collect_params())
        for epoch in range(epochs):
            start = time.time()
            for _ in range(10):
                with mx.autograd.record():
                    score_train = model(g_train, src_train, dst_train)
                    loss_train = cross_entropy(score_train, rating_train).mean()
                    # print(cross_entropy(score_train, rating_train))
                    # [0.691183   0.6853656  0.69425696... 0.6956804  0.7064479  0.6994275]
                    # < NDArray18806 @ cpu(0) >
                    # print(cross_entropy(score_train, rating_train).mean())
                    # [0.68960416]
                    # < NDArray 1 @ cpu(0) >
                    loss_train.backward()
                trainer.step(1)

            h_val = model.encoder(g)
            # print(h_val)
            score_val = model.decoder(h_val[src_test], h_val[dst_test])
            # print(h_val[src_test])
            # print(h_val[dst_test])
            # np.set_printoptions(threshold=np.inf)
            # print(score_val)
            loss_val = cross_entropy(score_val, rating_test).mean()
            train_auc = metrics.roc_auc_score(np.squeeze(rating_train.asnumpy()), np.squeeze(score_train.asnumpy()))
            val_auc = metrics.roc_auc_score(np.squeeze(rating_test.asnumpy()), np.squeeze(score_val.asnumpy()))
            results_val = [0 if j < 0.5 else 1 for j in np.squeeze(score_val.asnumpy())]
            # print(results_val)
            accuracy_val = metrics.accuracy_score(rating_test.asnumpy(), results_val)
            precision_val = metrics.precision_score(rating_test.asnumpy(), results_val)
            recall_val = metrics.recall_score(rating_test.asnumpy(), results_val)
            f1_val = metrics.f1_score(rating_test.asnumpy(), results_val)
            end = time.time()

            print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss_train.asscalar(),
                  'Val Loss: %.4f' % loss_val.asscalar(),
                  'Acc: %.4f' % accuracy_val, 'Pre: %.4f' % precision_val, 'Recall: %.4f' % recall_val,
                  'F1: %.4f' % f1_val, 'Train AUC: %.4f' % train_auc, 'Val AUC: %.4f' % val_auc,
                  'Time: %.2f' % (end - start))

        h_test = model.encoder(g)
        score_test = model.decoder(h_test[src_test], h_test[dst_test])
        fpr, tpr, thresholds = metrics.roc_curve(np.squeeze(rating_test.asnumpy()), np.squeeze(score_test.asnumpy()))
        test_auc = metrics.auc(fpr, tpr)
        results_test = [0 if j < 0.5 else 1 for j in np.squeeze(score_test.asnumpy())]
        accuracy_test = metrics.accuracy_score(rating_test.asnumpy(), results_test)
        precision_test = metrics.precision_score(rating_test.asnumpy(), results_test)
        recall_test = metrics.recall_score(rating_test.asnumpy(), results_test)
        f1_test = metrics.f1_score(rating_test.asnumpy(), results_test)

        print('Fold:', i + 1, 'Test Acc: %.4f' % accuracy_test, 'Test Pre: %.4f' % precision_test,
              'Test Recall: %.4f' % recall_test, 'Test F1: %.4f' % f1_test, 'Test AUC: %.4f' % test_auc)

        auc_result.append(test_auc)
        acc_result.append(accuracy_test)
        pre_result.append(precision_test)
        recall_result.append(recall_test)
        f1_result.append(f1_test)

        fprs.append(fpr)
        tprs.append(tpr)
    print('## Training Finished !')
    print('----------------------------------------------------------------------------------------------------------')

    return auc_result, acc_result, pre_result, recall_result, f1_result, fprs, tprs