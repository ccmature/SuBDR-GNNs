import argparse
import numpy as np
import torch
import os
from gcn import GCN
from deeprobust.graph.utils import preprocess
from subdrgnn import BdrGNN
import pdb
import time
import scipy.sparse as sp


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.02, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.03, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.8, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora',  'CiteSeer','Photo','PolBlogs', 'cora_ml'], help='dataset')
parser.add_argument('--epochs', type=int,  default=100, help='Number of epochs to train.')
parser.add_argument('--gamma', type=float, default=0.1, help='alpha')
parser.add_argument('--inner_steps', type=int, default=10, help='steps for inner optimization')
parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
parser.add_argument('--lr_adj', type=float, default=0.02, help='lr for training adj')
parser.add_argument('--ptb_rate', type=float, default=0, help="noise ptb_rate")
parser.add_argument('--lambda_', type=float, default=5, help='lambda')
parser.add_argument('--lambda__', type=float, default=0.01, help='beta')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")

print(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


path = 'data/' + args.dataset + '/'

perturbed_adj = torch.load(os.path.join(path, f'adj_{args.ptb_rate}.pt'))


features = torch.load(path + 'features.pt')
labels = torch.load(path + 'labels.pt')

adj, features, labels = preprocess(perturbed_adj, features, labels, preprocess_adj=False, preprocess_feature=False, device=device)

idx_train = torch.load(os.path.join(path, f'idx_train.pt'), weights_only=False)
idx_val = torch.load(os.path.join(path, f'idx_val.pt'), weights_only=False)
idx_test = torch.load(os.path.join(path, f'idx_test.pt'), weights_only=False)
idx_train = torch.LongTensor(idx_train).to(device)
idx_val = torch.LongTensor(idx_val).to(device)
idx_test = torch.LongTensor(idx_test).to(device)

train_labels = labels[idx_train]

label_diff_mask = train_labels.unsqueeze(1) != train_labels.unsqueeze(0)

adj[torch.tensor(idx_train).unsqueeze(1), torch.tensor(idx_train).unsqueeze(0)] = adj[torch.tensor(idx_train).unsqueeze(1), torch.tensor(idx_train).unsqueeze(0)] * (~label_diff_mask)

acc = np.zeros([10])
time_ = np.zeros([10])
for i in range(10):
    # 每次循环生成不同的随机种子
    current_seed = args.seed + i
    np.random.seed(current_seed)
    torch.manual_seed(current_seed)
    if args.cuda:
        torch.cuda.manual_seed(current_seed)

    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout, device=device)

    t1 = time.time()
    brdgnn = BdrGNN(model, args, device)
    brdgnn.fit(features, adj, labels, idx_train, idx_val)
    acc[i] = brdgnn.test(adj, features, labels, idx_test)
    time_[i] = time.time() - t1
    print(acc[i], time_[i])

# pdb.set_trace()
print(np.mean(acc), np.std(acc))
print(time_, np.mean(time_))
print(acc)