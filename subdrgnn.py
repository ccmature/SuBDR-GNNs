import pdb
import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.utils import accuracy
from deeprobust.graph import utils


class BdrGNN:

    def __init__(self, model, args, device):
        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_graph = None
        self.weights = None
        self.estimator = None
        self.best_s = None
        self.best_b = None
        self.model = model.to(device)


    def fit(self, features, adj, labels, idx_train, idx_val, **kwargs):

        args = self.args
        self.k = torch.max(labels) + 1

        self.M = torch.zeros_like(adj).to(self.device)

        self.train_data = (labels[:, None] == labels[None, :]).float()
        num_nodes = adj.shape[0]
        self.indicator = torch.zeros((num_nodes, num_nodes), dtype=torch.bool).to(self.device)
        self.indicator[idx_train[:, None], idx_train] = True


        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        estimator = EstimateAdj(adj, device=self.device).to(self.device)
        self.estimator = estimator
        self.optimizer_adj = optim.SGD(estimator.parameters(), momentum=0.9, lr=args.lr_adj)

        self.W1 = torch.zeros_like(adj).to(self.device)
        self.W2 = torch.zeros_like(adj).to(self.device)

        self.S = adj.to(self.device)

        # Train model
        for epoch in range(args.epochs):

            for i in range(int(args.outer_steps)):
                self.train_adj(epoch, features, adj, labels,
                               idx_train, idx_val)

            for i in range(int(args.inner_steps)):
                self.train_gcn(epoch, features, self.estimator.estimated_adj,
                               labels, idx_train, idx_val)

            with torch.no_grad():
                lambda_ = 8e-5
                D, V = torch.linalg.eigh(self.L + lambda_ * torch.eye(self.L.size(0)).to(self.device))
                D = D.real
                V = V.real

                _, ind = torch.sort(D)

                V_k = V[:, ind[:self.k]]

                self.W1 = (V_k @ V_k.T).to(self.device)

                del D, V, V_k


        print("Optimization Finished!")
        print(args)

        # Testing
        print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)

    def train_gcn(self, epoch, features, adj, labels, idx_train, idx_val):
        args = self.args
        estimator = self.estimator
        adj = estimator.normalize()

        # adj = (adj + self.norm_llm_adj) / 2

        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        _, output = self.model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        self.optimizer.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        _, output = self.model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

        # update = False

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = adj.detach()
            self.best_b = estimator.estimated_adj.detach()
            self.best_s = self.S.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print('\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())
            # update = True

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = adj.detach()
            self.best_b = estimator.estimated_adj.detach()
            self.best_s = self.S.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())
        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()),
                      'time: {:.4f}s'.format(time.time() - t))
    # 优化Z和S
    def train_adj(self, epoch, features, adj, labels, idx_train, idx_val):
        args = self.args
        estimator = self.estimator
        estimator.train()
        self.optimizer_adj.zero_grad()

        normalized_adj = estimator.normalize()

        x, output = self.model(features, normalized_adj)

        loss_gcn = F.nll_loss(output[idx_train], labels[idx_train])
        loss = loss_gcn + args.lambda_ * torch.sum((self.S - estimator.estimated_adj) ** 2)

        acc_train = accuracy(output[idx_train], labels[idx_train])

        loss.backward()

        self.optimizer_adj.step()

        z = self.estimator.estimated_adj
        with torch.no_grad():
            x = F.normalize(x, p=2, dim=1)

            diff = x.unsqueeze(1) - x.unsqueeze(0)
            euclidean_distances = torch.norm(diff, p=2, dim=2)
            sigma = 1
            similarity_matrix = torch.exp(-euclidean_distances ** 2 / (2 * sigma ** 2))

        self.M = torch.where(self.indicator, self.train_data, similarity_matrix)

        self.M = 1 - self.M

        # lambda_ = lambda, lambda__ = beta, gamma = alpha
        if epoch < 5:
            lambda__ = 0
        else:
            lambda__ = args.lambda__

        data = z - (lambda__ / (2 * args.lambda_)) * self.M - (args.gamma / (2 * args.lambda_)) * (torch.diag(self.W1).unsqueeze(1).repeat(1, self.W1.size(1)) - self.W1)

        data = torch.clamp((data + data.T) / 2, min=0)

        self.S = data

        one = torch.ones(data.size(0)).to(self.device)
        self.L = torch.diag(data @ one) - data

        self.model.eval()
        normalized_adj = estimator.normalize()

        _, output = self.model(features, normalized_adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        if epoch % 1 == 0:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()))
            # 'time: {:.4f}s'.format(time.time() - t))

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = normalized_adj.detach()
            self.best_b = estimator.estimated_adj.detach()
            self.best_s = self.S.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = normalized_adj.detach()
            self.best_b = estimator.estimated_adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            self.best_s = self.S.detach()
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())

    def test(self, adj, features, labels, idx_test):
        """Evaluate the performance of ProGNN on test set
        """
        print("\t=== testing ===")
        self.model.eval()
        adj = self.best_graph
        if self.best_graph is None:
            adj = self.estimator.normalize()

        
        _, output = self.model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])

        print("\tTest set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()


class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, adj, symmetric=False, device='cpu'):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self._init_estimation(adj)
        self.symmetric = symmetric
        self.device = device
        self.ori = adj

    def _init_estimation(self, adj):
        with torch.no_grad():
            n = len(adj)
            self.estimated_adj.data.copy_(torch.zeros_like(adj))

    def forward(self):
        return self.estimated_adj

    def normalize(self):
        adj = self.estimated_adj * self.ori

        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx