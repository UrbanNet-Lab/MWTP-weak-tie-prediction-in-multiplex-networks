import pickle as pkl
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
from model.MWTPModel import MWTPModel
from args import parse_args
from utils import sample_test_batch
from sample import iterate_return
import time


def get_weak_info(datsetname,idx_test,layer_num,num_nodes):
    ''' search weak ties'''
    print('Start generate strong ties and weak ties.')
    weak_dic = pkl.load(open('./community res/louvain/{}-dic.pkl'.format(datsetname), "rb"))
    wdic = weak_dic['dic_all']
    idx_test_strong,idx_test_strong_full, idx_test_weak,idx_test_weak_full = [], [], [], []
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

        idx_test_strong_full.append([strong_true_edges + strong_false_edges,len(strong_true_edges),len(strong_false_edges)])
        idx_test_weak_full.append([weak_true_edges + weak_false_edges,len(weak_true_edges),len(weak_false_edges)])

        min_weak_num = min(len(weak_true_edges), len(weak_false_edges))
        weak_true_edges = random.sample(weak_true_edges, min_weak_num)
        weak_false_edges = random.sample(weak_false_edges, min_weak_num)
        min_strong_num = min(len(strong_true_edges), len(strong_false_edges))
        strong_true_edges = random.sample(strong_true_edges, min_strong_num)
        strong_false_edges = random.sample(strong_false_edges, min_strong_num)
        print(
            f'after deal situation len(strong_true_edges):{len(strong_true_edges)}, len(strong_false_edges):{len(strong_false_edges)}, len(weak_true_edges):{len(weak_true_edges)}, len(weak_false_edges):{len(weak_false_edges)}')
        print('======================================')
        idx_test_strong.append([strong_true_edges + strong_false_edges,len(strong_true_edges),len(strong_false_edges)])
        idx_test_weak.append([weak_true_edges + weak_false_edges,len(weak_true_edges),len(weak_false_edges)])
    print('Generate strong ties and weak ties done!')
    return idx_test_strong,idx_test_strong_full, idx_test_weak,idx_test_weak_full

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
    eval_data = get_weak_info(args.datasetname, idx_test, layer_num, num_nodes)
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

    model = MWTPModel(layer_num, num_nodes, feats, adj_lists, args.emb_dim,device)

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
        model.load_state_dict(torch.load(f'checkpoint/{args.datasetname}_{times}_model_checkpoint.pth'))
        eval(model, idx_test, eval_data, ori_labels, ori_adj)
        return


    print('Start Train Step')
    return_list_all = []
    for l in range(layer_num):
        return_list_all.append(iterate_return(idx_train[l], [10, 10], labels[l], break_adj[l], args.batch_num))

    '''重点修改'''
    entity_dict = {}
    for l in range(layer_num):
        entity_dict[f'{l}'] = len(return_list_all[l])
    entity_dict = sorted(entity_dict.items(), key=lambda x: x[1])
    entity_dict_sort = [int(x) for x in list(dict(entity_dict).keys())]
    '''20241222'''
    entity_list_sort = [len(return_list_all[x]) for x in entity_dict_sort]
    ''''''

    auc_patience = 0
    best_auc = -float('inf')
    for epoch in range(args.epoches):
        model.train()
        epoches_times, epoches_loss, epoches_loss_val, epoches_acc, epoches_acc_val, epoches_auc, epoches_ap, epoches_f1 = [], [], [], [], [], [], [], []
        current_interval = 0
        for kia in tqdm(range(len(return_list_all[entity_dict_sort[-1]])),desc='Processing'):
            loss, val_loss, acc, auc, val_acc, ap, f1 = [], [], [], [], [], [], []
            # 如果当前 i 已经超过当前区间的上限，跳到下一个区间
            while kia+1 > entity_list_sort[current_interval]:
                current_interval += 1
            start_time = time.time()
            model.optimizer.zero_grad()
            for train_layer in entity_dict_sort[current_interval:]:
                train_nodes = return_list_all[train_layer][kia][0]
                train_targets = return_list_all[train_layer][kia][1]
                val_nodes = return_list_all[train_layer][kia][2]
                val_targets = return_list_all[train_layer][kia][3]
                # tttt = time.time()
                loss_t, acc_t, auc_t, ap_t, f1_t = model(train_nodes, train_targets, train_layer)
                # print('time train sets:',time.time()-tttt)
                # tttt2 = time.time()
                val_loss_t, val_acc_t = model(val_nodes, val_targets, train_layer, run_type='valid')
                # print('time val sets:',time.time()-tttt2)
                loss.append(loss_t)
                val_loss.append(val_loss_t)
                acc.append(acc_t)
                auc.append(auc_t)
                val_acc.append(val_acc_t)
                ap.append(ap_t)
                f1.append(f1_t)
            loss = sum(loss) / len(loss)
            val_loss = sum(val_loss) / len(val_loss)
            acc = sum(acc) / len(acc)
            auc = sum(auc) / len(auc)
            val_acc = sum(val_acc) / len(val_acc)
            ap = sum(ap) / len(ap)
            f1 = sum(f1) / len(f1)
            loss.backward()
            model.optimizer.step()
            end_time = time.time()
            epoches_times.append(end_time - start_time)
            epoches_loss.append(loss)
            epoches_loss_val.append(val_loss)
            epoches_acc.append(acc)
            epoches_acc_val.append(val_acc)
            epoches_auc.append(auc)
            epoches_ap.append(ap)
            epoches_f1.append(f1)
        print("epoches: {}/{} | ".format(epoch + 1, args.epoches),
              "train loss: {:.4f} | ".format(sum(epoches_loss) / len(epoches_loss)),
              "train acc: {:.4f} | ".format(sum(epoches_acc) / len(epoches_acc)),
              "train ap: {:.4f} | ".format(sum(epoches_ap) / len(epoches_ap)),
              "train f1: {:.4f} | ".format(sum(epoches_f1) / len(epoches_f1)),
              "train auc: {:.4f} | ".format(sum(epoches_auc) / len(epoches_auc)),
              "val loss: {:.4f} | ".format(sum(epoches_loss_val) / len(epoches_loss_val)),
              "val acc: {:.4f} | ".format(sum(epoches_acc_val) / len(epoches_acc_val)),
              "time: {:.4f} | ".format(sum(epoches_times)))
        '''2023/3/24 end'''
        if (epoch+1)%2 == 0:
            patience_auc = eval(model, idx_test, eval_data, ori_labels, ori_adj)
            if patience_auc > best_auc:
                auc_patience = 0
                best_auc = patience_auc
            else:
                auc_patience += 1
                if auc_patience > args.patience:
                    print("Early Stopping")
                    break
    patience_auc = eval(model, idx_test, eval_data, ori_labels, ori_adj)
    if args.save_checkpoint:
        torch.save(model.state_dict(), f'checkpoint/{args.datasetname}_{times}_model_checkpoint.pth')
        print("Model parameters saved!")


def eval(model,idx_test, eval_data,ori_labels,ori_adj):
    print('Start Eval Step')
    model.eval()
    with torch.no_grad():
        idx_test_strong,idx_test_strong_full, idx_test_weak,idx_test_weak_full = eval_data
        eval_start_time = time.time()
        test_Auc = test(model,idx_test,ori_labels,ori_adj,test_type='Overall Ties')
        print('Overall layer overall edges test time:', time.time() - eval_start_time)
        test(model, idx_test_strong, ori_labels, ori_adj,test_type='Strong Ties')
        test(model, idx_test_strong_full, ori_labels, ori_adj,test_type='Strong Full Ties')
        test(model, idx_test_weak, ori_labels, ori_adj,test_type='Weak Ties')
        test(model, idx_test_weak_full, ori_labels, ori_adj,test_type='Weak Full Ties')
    return test_Auc

def test(model,idx_test,ori_labels,ori_adj,test_type=''):
    test_loss, test_acc, test_Auc, test_ap, test_f1 = [], [], [], [], []
    for kv in range(layer_num):
        if test_type != 'Overall Ties' and (idx_test[kv][1] == 0 or idx_test[kv][2] == 0):
            continue
        test_loss_p, test_acc_p, test_Auc_p, test_ap_p, test_f1_p = [], [], [], [], []
        if test_type == 'Overall Ties':
            test_data_pair = sample_test_batch(idx_test[kv], ori_labels[kv], ori_adj[kv])
        else:
            test_data_pair = sample_test_batch(idx_test[kv][0], ori_labels[kv], ori_adj[kv])
        num_batch = len(test_data_pair)
        if test_type == 'Overall Ties':
            t_layer_time = time.time()
        for test_batch in test_data_pair:
            test_node, test_targets = test_batch
            test_loss_t, test_acc_t, test_Auc_t, test_ap_t, test_f1_t = model(test_node, test_targets,kv)
            test_loss_p.append(test_loss_t)
            test_acc_p.append(test_acc_t)
            test_Auc_p.append(test_Auc_t)
            test_ap_p.append(test_ap_t)
            test_f1_p.append(test_f1_t)
        if test_type == 'Overall Ties':
            print(f'Layer {kv}, time: {time.time()-t_layer_time}')
        test_loss.append(sum(test_loss_p) / num_batch)
        test_acc.append(sum(test_acc_p) / num_batch)
        test_Auc.append(sum(test_Auc_p) / num_batch)
        test_ap.append(sum(test_ap_p) / num_batch)
        test_f1.append(sum(test_f1_p) / num_batch)
    test_loss = sum(test_loss) / len(test_loss)
    test_acc = sum(test_acc) / len(test_acc)
    test_Auc = sum(test_Auc) / len(test_Auc)
    test_ap = sum(test_ap) / len(test_ap)
    test_f1 = sum(test_f1) / len(test_f1)
    print("{} ---- loss: {:.4f} | acc: {:.4f} | ap: {:.4f} | f1: {:.4f} | auc: {:.4f}".format(test_type,test_loss, test_acc, test_ap, test_f1, test_Auc))
    if test_type == 'Overall Ties':
        return test_Auc

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
