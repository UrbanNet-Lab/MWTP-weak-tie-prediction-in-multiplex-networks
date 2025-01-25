import pickle as pkl
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataloder import construct_loader
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
from model.MWTPModel import MWTPModel
from args import parse_args
import time
from itertools import zip_longest

def get_weak_info(datsetname, idx_test, labels, layer_num, num_nodes):
    ''' search weak ties'''
    print('Start generate strong ties and weak ties.')
    weak_dic = pkl.load(open('./community res/louvain/{}-dic.pkl'.format(datsetname), "rb"))
    wdic = weak_dic['dic_all']
    idx_test_strong,idx_test_strong_full, idx_test_weak,idx_test_weak_full = [], [], [], []
    idx_test_slen,idx_test_sflen, idx_test_wlen,idx_test_wflen = [], [], [], []
    for l in range(layer_num):
        print(
            f'idx_train len: {len(idx_train[l])}  |  idx_val len: {len(idx_val[l])}  |  idx_test len: {len(idx_test[l])}')
        strong_true_edges, strong_false_edges, weak_true_edges, weak_false_edges = [], [], [], []
        for item in idx_test[l]:
            e1, e2 = divmod(item, num_nodes)
            if wdic[l][int(e1)] != wdic[l][int(e2)]:
                if ori_adj[l][int(e1)][int(e2)] == 1:
                    weak_true_edges.append(item)
                else:
                    weak_false_edges.append(item)
            else:
                if ori_adj[l][int(e1)][int(e2)] == 1:
                    strong_true_edges.append(item)
                else:
                    strong_false_edges.append(item)
        print(
            f'init situation len(strong_true_edges):{len(strong_true_edges)}, len(strong_false_edges):{len(strong_false_edges)}, len(weak_true_edges):{len(weak_true_edges)}, len(weak_false_edges):{len(weak_false_edges)}')
        if datsetname == 'acm' and l == 1:
            # 第二层的强负边全部来自弱负边 77436 0 143640 221076
            np.random.shuffle(weak_false_edges)
            strong_false_edges = weak_false_edges[:len(strong_true_edges)]
            weak_false_edges = weak_false_edges[len(strong_true_edges):]
        if (datsetname == 'amazon' and l == 2) or (datsetname == 'imdb' and l == 0):
            # amazon 第三层网络的 正边都是强正边，负边都是弱负边
            # imdb 第一层 正边都是强正边，负边都是弱负边
            strong_false_edges = weak_false_edges
            weak_true_edges, weak_false_edges = [], []
        sf_t = strong_true_edges + strong_false_edges
        random.shuffle(sf_t)
        idx_test_strong_full.append(sf_t)
        wf_t = weak_true_edges + weak_false_edges
        random.shuffle(wf_t)
        idx_test_weak_full.append(wf_t)
        idx_test_sflen.append([len(strong_true_edges),len(strong_false_edges)])
        idx_test_wflen.append([len(weak_true_edges),len(weak_false_edges)])


        min_weak_num = min(len(weak_true_edges), len(weak_false_edges))
        weak_true_edges = random.sample(weak_true_edges, min_weak_num)
        weak_false_edges = random.sample(weak_false_edges, min_weak_num)
        min_strong_num = min(len(strong_true_edges), len(strong_false_edges))
        strong_true_edges = random.sample(strong_true_edges, min_strong_num)
        strong_false_edges = random.sample(strong_false_edges, min_strong_num)
        print(
            f'after deal situation len(strong_true_edges):{len(strong_true_edges)}, len(strong_false_edges):{len(strong_false_edges)}, len(weak_true_edges):{len(weak_true_edges)}, len(weak_false_edges):{len(weak_false_edges)}')
        print('======================================')
        s_t = strong_true_edges + strong_false_edges
        random.shuffle(s_t)
        idx_test_strong.append(s_t)
        w_t = weak_true_edges + weak_false_edges
        random.shuffle(w_t)
        idx_test_weak.append(w_t)
        idx_test_slen.append([len(strong_true_edges), len(strong_false_edges)])
        idx_test_wlen.append([len(weak_true_edges), len(weak_false_edges)])
    idx_test_strong_loaders, max_len_s = construct_loader(idx_test_strong, labels, layer_num, num_nodes, 1024)
    idx_test_strong_fullloaders, max_len_sf = construct_loader(idx_test_strong_full, labels, layer_num, num_nodes, 1024)
    idx_test_weak_loaders, max_len_w = construct_loader(idx_test_weak, labels, layer_num, num_nodes, 1024)
    idx_test_weak_full_loaders, max_len_wf = construct_loader(idx_test_weak_full, labels, layer_num, num_nodes, 1024)
    print('Generate strong ties and weak ties done!')
    return [[idx_test_strong_loaders, max_len_s],[idx_test_strong_fullloaders, max_len_sf],\
           [idx_test_weak_loaders, max_len_w],[idx_test_weak_full_loaders, max_len_wf]],\
           [idx_test_slen, idx_test_sflen, idx_test_wlen, idx_test_wflen]

def get_dataset(dataname='ckm'):
    # data_get = pkl.load(open('./data_pkl/{}.pkl'.format(dataname), "rb"))
    data_get = np.load(f'./data_pkl/{dataname}.npz')
    ori_adj, break_adj, feats = data_get['ori_adj'], data_get['adj'], data_get['feat_data']
    if dataname == 'ckm' or dataname == 'amazon':
        idx_train = [data_get['idx_train'], data_get['idx_train1'], data_get['idx_train2']]
        idx_val = [data_get['idx_val'], data_get['idx_val1'], data_get['idx_val2']]
        idx_test = [data_get['idx_test'], data_get['idx_test1'], data_get['idx_test2']]
    elif dataname == 'imdb' or dataname == 'acm':
        idx_train = [data_get['idx_train'], data_get['idx_train1']]
        idx_val = [data_get['idx_val'], data_get['idx_val1']]
        idx_test = [data_get['idx_test'], data_get['idx_test1']]
    return ori_adj, break_adj, feats, idx_train, idx_val, idx_test

def train(ori_adj, break_adj, feats, idx_train, idx_test, layer_num, num_nodes, args,device,times):
    labels = []
    ori_labels = []
    for l in range(layer_num):
        labels.append(torch.Tensor(break_adj[l].flatten()))
        ori_labels.append(torch.Tensor(ori_adj[l].flatten()))
    adj_lists = []
    for ki in range(layer_num):
        adj_lists_temp = defaultdict(set)
        [row, col] = np.where(break_adj[ki] == 1)
        for i in range(row.size):
            adj_lists_temp[row[i]].add(col[i])
            adj_lists_temp[col[i]].add(row[i])
        adj_lists.append(adj_lists_temp)
    eval_data = get_weak_info(args.datasetname, idx_test, ori_labels, layer_num, num_nodes)

    model = MWTPModel(layer_num, num_nodes, feats, adj_lists, args.emb_dim,device,args.inter_aggregation)

    # Print all trainable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter name: {name}, Shape: {param.shape}")
    # Count total and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    if args.run_test:
        model.load_state_dict(torch.load(f'checkpoint/{args.inter_aggregation}_{args.datasetname}_{times}_model_checkpoint.pth'))
        eval(model, idx_test, eval_data, ori_labels, ori_adj)
        return


    print('Start Train Step')
    train_data_loaders, max_len_train = construct_loader(idx_train, labels, layer_num, num_nodes, args.batch_num)
    val_data_loaders, max_len_val= construct_loader(idx_val, labels, layer_num, num_nodes, args.batch_num)
    test_data_loaders, max_len_test = construct_loader(idx_test, ori_labels, layer_num, num_nodes, args.batch_num)

    auc_patience = 0
    best_auc = -float('inf')
    for epoch in range(args.epoches):
        # import cProfile
        # import pstats
        # profiler = cProfile.Profile()
        # profiler.enable()

        model.train()
        epoches_times, epoches_loss, epoches_acc, epoches_auc, epoches_ap, epoches_f1 = [], [], [], [], [], []
        for batch_idx, *batch_data in tqdm(enumerate(zip_longest(*train_data_loaders, fillvalue=None)),
                                           desc='processing',unit='items',total=max_len_train):
            # import cProfile
            # import pstats
            # profiler = cProfile.Profile()
            # profiler.enable()

            loss, acc, auc, ap, f1 = [], [], [], [], []
            start_train_time = time.time()
            for train_layer, train_data in enumerate(*batch_data):
                if train_data is None:
                    continue
                train_nodes, train_targets = train_data
                loss_t, acc_t, auc_t, ap_t, f1_t = model(train_nodes.view(-1), train_targets, train_layer)
                loss.append(loss_t)
                acc.append(acc_t)
                auc.append(auc_t)
                ap.append(ap_t)
                f1.append(f1_t)

            if len(loss) != 0:
                loss = sum(loss) / len(loss)
                model.optimizer.zero_grad()
                loss.backward()
                model.optimizer.step()
                epoches_times.append(time.time() - start_train_time)
                epoches_loss.append(loss)
                epoches_acc.append(sum(acc) / len(acc))
                epoches_auc.append(sum(auc) / len(auc))
                epoches_ap.append(sum(ap) / len(ap))
                epoches_f1.append(sum(f1) / len(f1))

            # profiler.disable()
            # stats = pstats.Stats(profiler)
            # stats.sort_stats('cumulative').print_stats()
            # sys.exit()
        print("train epoches: {}/{} | ".format(epoch + 1, args.epoches),
              "loss: {:.4f} | ".format(sum(epoches_loss) / len(epoches_loss)),
              "acc: {:.4f} | ".format(sum(epoches_acc) / len(epoches_acc)),
              "ap: {:.4f} | ".format(sum(epoches_ap) / len(epoches_ap)),
              "f1: {:.4f} | ".format(sum(epoches_f1) / len(epoches_f1)),
              "auc: {:.4f} | ".format(sum(epoches_auc) / len(epoches_auc)),
              "time: {:.4f}".format(sum(epoches_times)))
        # profiler.disable()
        # stats = pstats.Stats(profiler)
        # stats.sort_stats('cumulative').print_stats()
        # sys.exit()
        eval(model, val_data_loaders, max_len_val, eval_type='eval')
        if (epoch + 1) % 2 == 0:
            with torch.no_grad():
                patience_auc = test(model, test_data_loaders, max_len_test, eval_data, eval_type='test')
            if patience_auc > best_auc:
                auc_patience = 0
                best_auc = patience_auc
            else:
                auc_patience += 1
                if auc_patience > args.patience:
                    print("Early Stopping")
                    break
        torch.cuda.empty_cache() # 每次计算后清理GPU缓存
    patience_auc = test(model, test_data_loaders, max_len_test, eval_data)
    if args.save_checkpoint:
        torch.save(model.state_dict(), f'checkpoint/{args.inter_aggregation}_{args.datasetname}_{times}_model_checkpoint.pth')
        print("Model parameters saved!")

def test(model, test_data_loaders, max_len_loader, eval_data, eval_type='test'):
    print('----------------------------Start Eval Step--------------------------------------')
    bathes_auc = eval(model, test_data_loaders, max_len_loader, eval_type='Overall Ties')
    data_loaders, data_info = eval_data
    idx_test_strong_loaders, idx_test_strong_full_loaders, idx_test_weak_loaders, idx_test_weak_full_loaders = data_loaders
    strong_info, strong_full_info, weak_info, weak_full_info = data_info
    eval(model, idx_test_strong_loaders[0], idx_test_strong_loaders[1], data_info=strong_info, eval_type='Strong Ties')
    eval(model, idx_test_strong_full_loaders[0], idx_test_strong_full_loaders[1], data_info=strong_full_info,  eval_type='Strong Full Ties')
    eval(model, idx_test_weak_loaders[0], idx_test_weak_loaders[1], data_info=weak_info,  eval_type='Weak Ties')
    eval(model, idx_test_weak_full_loaders[0], idx_test_weak_full_loaders[1], data_info=weak_full_info,  eval_type='Weak Full Ties')
    print('----------------------------End Eval Step--------------------------------------')
    if eval_type == 'test':
        return bathes_auc
def eval(model, data_loaders, max_len_loader, data_info=None, eval_type=''):
    model.eval()
    with torch.no_grad():
        batches_times, batches_loss, batches_acc, batches_auc, batches_ap, batches_f1 = [], [], [], [], [], []
        for batch_idx, *batch_data in enumerate(zip_longest(*data_loaders, fillvalue=None)):#,
                                           # desc='processing', unit='items', total=max_len_loader):
            loss, acc, auc, ap, f1 = [], [], [], [], []
            start_train_time = time.time()
            for eval_layer, eval_data in enumerate(*batch_data):
                if eval_data is None:
                    continue
                if data_info and (data_info[eval_layer][0] == 0 or data_info[eval_layer][1] == 0):
                    continue
                eval_nodes, eval_targets = eval_data
                loss_t, acc_t, auc_t, ap_t, f1_t = model(eval_nodes.view(-1), eval_targets, eval_layer)
                loss.append(loss_t)
                acc.append(acc_t)
                auc.append(auc_t)
                ap.append(ap_t)
                f1.append(f1_t)

            if len(loss) != 0:
                batches_times.append(time.time() - start_train_time)
                batches_loss.append(sum(loss) / len(loss))
                batches_acc.append(sum(acc) / len(acc))
                batches_auc.append(sum(auc) / len(auc))
                batches_ap.append(sum(ap) / len(ap))
                batches_f1.append(sum(f1) / len(f1))
        print("{} loss: {:.4f} | ".format(eval_type,sum(batches_loss) / len(batches_loss)),
              "acc: {:.4f} | ".format(sum(batches_acc) / len(batches_acc)),
              "ap: {:.4f} | ".format(sum(batches_ap) / len(batches_ap)),
              "f1: {:.4f} | ".format(sum(batches_f1) / len(batches_f1)),
              "auc: {:.4f} | ".format(sum(batches_auc) / len(batches_auc)),
              "time: {:.4f}".format(sum(batches_times)))
    if eval_type == 'Overall Ties':
        return round(sum(batches_auc) / len(batches_auc),4)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    if args.device == 'gpu':
        cuda = torch.cuda.is_available()
        device = torch.device("cuda:0") if cuda else torch.device("cpu")
    elif args.device == 'cpu':
        device = torch.device("cpu")
    else:
        raise Exception("Invalid device option. Please choose 'gpu' or 'cpu'.")
    ori_adj, break_adj, feats, idx_train, idx_val, idx_test = get_dataset(args.datasetname)
    layer_num = ori_adj.shape[0]
    num_nodes = ori_adj.shape[1]
    print('adj shape: ', ori_adj.shape)
    print('layer num: ', layer_num)
    print('nodes num: ', num_nodes)
    for times in range(args.run_times):
        print(f'The times: {times}')
        train(ori_adj, break_adj, feats, idx_train, idx_test, layer_num, num_nodes, args, device,times)
        print('----------------------------------------------------------------------------------')



