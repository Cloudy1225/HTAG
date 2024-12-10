import argparse
import csv
import time
import datetime
import pickle

import dgl
import torch
import numpy as np
from dataloader import load_data
from model import GCN, SAGE, GAT
from train_and_test import train_gnn, test_gnn
from utils import set_seed, get_training_config, get_logger, check_writable

import warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch DGL implementation')

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--eval_interval', type=int, default=1,
        help='Evaluate once per how many epochs'
    )
    parser.add_argument(
        "--output_path", type=str, default="outputs",
        help="Path to save outputs"
    )
    parser.add_argument(
        '--num_exp', type=int, default=5,
        help='Repeat how many experiments'
    )

    """Dataset"""
    parser.add_argument(
        '--dataset', type=str, default='TMDB',
        choices=['TMDB', 'CroVal', 'ArXiv', 'Book', 'DBLP', 'Patent']
    )

    """Model"""
    parser.add_argument(
        '--model_config_path',
        type=str,
        default='./train.conf.yaml',
        help='Path to model configuration'
    )
    parser.add_argument(
        '--model', type=str, default='SAGE',
        choices=['GCN', 'SAGE', 'GAT']
    )
    parser.add_argument(
        '--activation', type=str, default='ReLU',
        choices=['ReLU', 'ELU', 'PReLU', 'Identity']
    )
    parser.add_argument(
        '--norm_type', type=str, default='none',
        choices=['none', 'batch', 'layer']
    )
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=10240)
    parser.add_argument(
        '--num_workers', type=int, default=0, help='Number of workers for sampler'
    )

    """Optimization"""
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.00)

    args = parser.parse_args()
    return args


def run(args):
    set_seed(args.seed)

    device = f'cuda:{args.device}' \
        if torch.cuda.is_available() else 'cpu'
    conf = get_training_config(args.dataset, args.model, config_path=args.model_config_path)
    conf = dict(args.__dict__, **conf)

    out_dir = f'./{args.output_path}/{args.dataset}/{args.model}'
    check_writable(out_dir, overwrite=False)
    log_path = f'{out_dir}/log.txt'
    save_path = f'{out_dir}/model.pt'

    logger = get_logger(log_path)

    logger.info(str(conf))


    """Load heterogeneous graph"""
    pkl_data = f'./{args.output_path}/{args.dataset}/data.pkl'
    try:
        with open(pkl_data, 'rb') as f:
            g, idx_train, idx_val, idx_test = pickle.load(f)
    except FileNotFoundError:
        g, (idx_train, idx_val, idx_test), generate_node_features = load_data(dataset=args.dataset)
        g = generate_node_features(g)
        with open(pkl_data, 'wb') as f:
            pickle.dump((g, idx_train, idx_val, idx_test), f)

    # g, (idx_train, idx_val, idx_test), generate_node_features = load_data(dataset=args.dataset)
    # g = generate_node_features(g)

    """Convert heterogeneous to homogeneous"""
    # Reorder train/val/test splits
    target = g.target
    start_idx = 0
    for ntype in g.ntypes:
        if ntype != target:
            start_idx += g.num_nodes(ntype)
        else:
            break
    arange = torch.arange(start_idx, start_idx+g.num_nodes(target))
    idx_train = arange[idx_train]
    idx_val = arange[idx_val]
    idx_test = arange[idx_test]

    # to homogeneous
    labels = g.ndata['label'][target]
    g = dgl.to_homogeneous(g, ndata=['feat'])
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)  # add self-loops

    # assign zero_labels to non-target nodes
    if len(labels.shape) == 2:
        zero_labels = torch.zeros((g.num_nodes(), labels.shape[1]), dtype=labels.dtype)
    else:
        zero_labels = torch.zeros(g.num_nodes(), dtype=labels.dtype)
    zero_labels[arange] = labels
    g.ndata['label'] = zero_labels

    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    num_layers = len(conf['hid_dims'])


    """Neighbor Sampler"""
    g.create_formats_()
    sampler = dgl.dataloading.NeighborSampler([conf['fan_out']]*num_layers,
                                              prefetch_node_feats=['feat'],
                                              prefetch_labels=['label'])
    train_dataloader = dgl.dataloading.DataLoader(
        g, idx_train, sampler,
        batch_size=batch_size,
        shuffle=True, drop_last=False,
        num_workers=num_workers)
    sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers)
    val_dataloader = dgl.dataloading.DataLoader(
        g, idx_val, sampler_eval,
        batch_size=batch_size,
        shuffle=False, drop_last=False,
        num_workers=num_workers)
    test_dataloader = dgl.dataloading.DataLoader(
        g, idx_test, sampler_eval,
        batch_size=batch_size,
        shuffle=False, drop_last=False,
        num_workers=num_workers)


    """Model Initialization"""
    in_dim = g.ndata['feat'].shape[1]
    if len(g.ndata['label'].shape) > 1:
        multilabel = True
        num_classes = g.ndata['label'].shape[-1]
    else:
        multilabel = False
        num_classes = int(max(g.ndata['label'])) + 1
    if args.model == 'GCN':
        model = GCN(in_dim, conf['hid_dims'], num_classes,
                     conf['dropout'], conf['activation'], conf['norm_type'])
    elif args.model == 'SAGE':
        model = SAGE(in_dim, conf['hid_dims'], num_classes,
                     conf['dropout'], conf['activation'], conf['norm_type'])
    elif args.model == 'GAT':
        model = GAT(in_dim, conf['hid_dims'], num_classes, conf['num_heads'], conf['dropout'],
                     conf['attn_drop'], conf['activation'], conf['norm_type'])
    else:
        model = SAGE(in_dim, conf['hid_dims'], num_classes,
                     conf['dropout'], conf['activation'], conf['norm_type'])
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                               lr=conf['learning_rate'], weight_decay=conf['weight_decay'])

    """Training and Evaluation"""
    best_epoch = 0
    best_score_val = 0
    training_start = time.time()
    for epoch in range(1, conf['epochs'] + 1):
        epoch_start = time.time()
        loss = train_gnn(model, train_dataloader, optimizer, device, multilabel)
        time_taken = str(datetime.timedelta(seconds=(time.time() - epoch_start)))
        if epoch % conf['eval_interval'] == 0:
            micro_val, macro_val = test_gnn(model, val_dataloader, device, multilabel)
            if best_score_val < micro_val:
                best_epoch = epoch
                best_score_val = micro_val
                torch.save(model.state_dict(), save_path)
            else:
                if epoch-best_epoch > conf['patience']:
                    break
            print(
                f'Epoch {epoch:04d} | Loss={loss:.4f} | '
                f'Val Micro={micro_val:.2f} Macro={macro_val:.2f} | Time {time_taken}'
            )

    model.load_state_dict(torch.load(save_path, weights_only=False))
    micro_test, macro_test = test_gnn(model, test_dataloader, device, multilabel)
    time_taken = str(datetime.timedelta(seconds=(time.time() - training_start)))
    logger.info(f'Best Epoch {best_epoch} | Test Micro={micro_test:.2f} '
                f'Macro={macro_test:.2f} | Total time taken {time_taken}')

    return micro_test, macro_test


if __name__ == "__main__":
    args = get_args()

    micros = []
    macros = []
    for seed in range(args.num_exp):
        args.seed = seed
        micro_test, macro_test = run(args)
        micros.append(micro_test)
        macros.append(macro_test)

    micro_mean, micro_std = np.mean(micros), np.std(micros)
    macro_mean, macro_std = np.mean(macros), np.std(macros)
    micro = f'{micro_mean:.2f}+-{micro_std:.2f}'
    macro = f'{macro_mean:.2f}+-{macro_std:.2f}'
    print(f'Micro={micro} Macro={macro}')

    with open(f'./{args.output_path}/{args.dataset}/results.csv', 'a',
              encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        row = [args.dataset, args.model, micro, macro, micros, macros]
        writer.writerow(row)
