import os
import pickle
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset


class TripletUniformPair(Dataset):
    def __init__(self, w, num_item, pair, args):
        self.neg_w = 1 - w
        self.num_item = num_item
        self.pair = pair
        self.batch_size = args.batch_size

    def __getitem__(self, idx):
        idx = np.random.choice(self.pair.shape[0], size=self.batch_size)
        u = self.pair[idx, 0]
        i = self.pair[idx, 1]
        j = torch.multinomial(self.neg_w[u, :], num_samples=1).squeeze()
        return u, i, j

    def __len__(self):
        return 10*len(self.pair)


class BPR(nn.Module):
    def __init__(self, user_size, item_size, args):
        super().__init__()
        self.W = nn.Parameter(torch.rand(user_size, args.dim))
        self.H = nn.Parameter(torch.rand(item_size, args.dim))

    def forward(self, u, i, j):
        x_ui = torch.mul(self.W[u, :], self.H[i, :]).sum(dim=1)
        x_uj = torch.mul(self.W[u, :], self.H[j, :]).sum(dim=1)
        x_uij = x_ui - x_uj
        log_prob = torch.log(torch.sigmoid(x_uij)).mean()
        return -log_prob


def precision_and_recall_k(user_emb, item_emb, train_w, test_w, klist):
    """Compute precision at k using GPU.

    Args:
        user_emb (torch.Tensor): embedding for user [user_num, dim]
        item_emb (torch.Tensor): embedding for item [item_num, dim]
        train_w (torch.Tensor): mask array for train record [user_num, item_num]
        test_w (torch.Tensor): mask array for test record [user_num, item_num]
        k (list(int)): list of k
    Returns:
        (torch.Tensor, torch.Tensor) Precision and recall at k
    """
    # Compute all pair of training and test record
    # Reason why do sigmoid is sigmoid and compress value into [0, 1]
    # And we are going to make useless value to zero to make smallest value
    result = torch.mm(user_emb, item_emb.t())
    result = torch.sigmoid(result)

    # Mask pred and true
    # test_pred represents both test and negative record
    test_pred_mask = 1 - (train_w)
    test_pred = test_pred_mask * result
    # test_true represents only test record indicator
    test_true_mask = test_w
    test_true = test_true_mask * result

    # Sort indice and get test_pred_topk
    _, test_indices = torch.topk(test_pred, dim=1, k=max(klist))
    precisions, recalls = [], []
    for k in klist:
        topk_mask = torch.zeros_like(test_pred)
        topk_mask.scatter_(dim=1,
                           index=test_indices[:, :k],
                           src=torch.tensor(1.0).cuda())
        test_pred_topk = topk_mask * test_pred

        # Compare which is not zero and equal with test_true, which means that
        # both is not excluded by mask and is true value
        acc_result = (test_pred_topk != 0) & (test_pred_topk == test_true)
        precisions.append(acc_result.sum().float() / (user_emb.shape[0] * k))
        recalls.append((acc_result.float().sum(dim=1) / test_w.sum(dim=1)).mean())
    return precisions, recalls


def main(args):
    # Load preprocess data
    with open(args.data, 'rb') as f:
        dataset = pickle.load(f)

        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_w, test_w = dataset['train_w'], dataset['test_w']
        train_pair = dataset['train_pair']
        print('Load complete')

    # Convert to tensor and move to GPU
    train_w = torch.tensor(train_w, dtype=torch.float).cuda()
    test_w = torch.tensor(test_w, dtype=torch.float).cuda()
    train_pair = torch.tensor(train_pair, dtype=torch.long).cuda()

    # Create dataset, model, optimizer
    dataset = TripletUniformPair(train_w, item_size, train_pair, args)
    model = BPR(user_size, item_size, args).cuda()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    # Training
    smooth_loss = 0
    for epoch in range(args.n_epochs):
        for idx, (u, i, j) in enumerate(iter(dataset)):
            optimizer.zero_grad()
            loss = model(u, i, j)
            loss.backward()
            optimizer.step()
            smooth_loss = smooth_loss*0.99 + loss*0.01
            if idx % args.print_every == (args.print_every - 1):
                print('loss: %.4f' % smooth_loss)
            if idx % args.eval_every == (args.eval_every - 1):
                plist, rlist = precision_and_recall_k(model.W.detach(),
                                                      model.H.detach(),
                                                      train_w,
                                                      test_w,
                                                      klist=[1, 5, 10])
                print('P@1: %.4f, P@5: %.4f P@10: %.4f, R@1: %.4f, R@5: %.4f, R@10: %.4f' % (plist[0], plist[1], plist[2], rlist[0], rlist[1], rlist[2]))
            if idx % args.save_every == (args.save_every - 1):
                dirname = os.path.dirname(os.path.abspath(args.model))
                os.makedirs(dirname, exist_ok=True)
                torch.save(model.state_dict(), args.model)


if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                        type=str,
                        default=os.path.join('preprocessed', 'bpr-movielens-1m.pickle'),
                        help="File path for data")
    # Model
    parser.add_argument('--dim',
                        type=int,
                        default=4,
                        help="Dimension for embedding")
    # Optimizer
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help="Learning rate")
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.000025,
                        help="Weight decay factor")
    # Training
    parser.add_argument('--n_epochs',
                        type=int,
                        default=10,
                        help="Number of epoch during training")
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help="Batch size in one iteration")
    parser.add_argument('--print_every',
                        type=int,
                        default=10,
                        help="Period for printing smoothing loss during training")
    parser.add_argument('--eval_every',
                        type=int,
                        default=1000,
                        help="Period for evaluating precision and recall during training")
    parser.add_argument('--save_every',
                        type=int,
                        default=1000,
                        help="Period for saving model during training")
    parser.add_argument('--model',
                        type=str,
                        default=os.path.join('output', 'bpr.pt'),
                        help="File path for model")
    args = parser.parse_args()
    main(args)
