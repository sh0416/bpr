import os
import pickle
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class TripletUniformPair(Dataset):
    def __init__(self, num_item, user_list, pair):
        self.num_item = num_item
        self.user_list = user_list
        self.pair = pair

    def __getitem__(self, idx):
        idx = np.random.randint(len(self.pair))
        u = self.pair[idx][0]
        i = self.pair[idx][1]
        j = np.random.randint(self.num_item)
        while j in self.user_list[u]:
            j = np.random.randint(self.num_item)
        return u, i, j

    def __len__(self):
        return 10*len(self.pair)


class BPR(nn.Module):
    def __init__(self, user_size, item_size, dim):
        super().__init__()
        self.W = nn.Parameter(torch.rand(user_size, dim))
        self.H = nn.Parameter(torch.rand(item_size, dim))

    def forward(self, u, i, j):
        x_ui = torch.mul(self.W[u, :], self.H[i, :]).sum(dim=1)
        x_uj = torch.mul(self.W[u, :], self.H[j, :]).sum(dim=1)
        x_uij = x_ui - x_uj
        log_prob = torch.log(torch.sigmoid(x_uij)).mean()
        return -log_prob


def precision_and_recall_k(user_emb, item_emb, train_user_list, test_user_list, klist, batch=512):
    """Compute precision at k using GPU.

    Args:
        user_emb (torch.Tensor): embedding for user [user_num, dim]
        item_emb (torch.Tensor): embedding for item [item_num, dim]
        train_user_list (list(set)):
        test_user_list (list(set)):
        k (list(int)):
    Returns:
        (torch.Tensor, torch.Tensor) Precision and recall at k
    """
    # Calculate max k value
    max_k = max(klist)

    # Compute all pair of training and test record
    result = None
    for i in range(0, user_emb.shape[0], batch):
        # Create already observed mask
        mask = user_emb.new_ones([min([batch, user_emb.shape[0]-i]), item_emb.shape[0]])
        for j in range(batch):
            if i+j >= user_emb.shape[0]:
                break
            mask[j].scatter_(dim=0, index=torch.tensor(list(train_user_list[i+j])).cuda(), value=torch.tensor(0.0).cuda())
        # Calculate prediction value
        cur_result = torch.mm(user_emb[i:i+min(batch, user_emb.shape[0]-i), :], item_emb.t())
        cur_result = torch.sigmoid(cur_result)
        # Make zero for already observed item
        cur_result = torch.mul(mask, cur_result)
        _, cur_result = torch.topk(cur_result, k=max_k, dim=1)
        result = cur_result if result is None else torch.cat((result, cur_result), dim=0)

    result = result.cpu()
    # Sort indice and get test_pred_topk
    precisions, recalls = [], []
    for k in klist:
        precision, recall = 0, 0
        for i in range(user_emb.shape[0]):
            test = test_user_list[i]
            pred = set(result[i, :k].numpy().tolist())
            val = len(test & pred)
            precision += val / k
            recall += val / len(test)
        precisions.append(precision / user_emb.shape[0])
        recalls.append(recall / user_emb.shape[0])
    return precisions, recalls


def main(args):
    # Load preprocess data
    with open(args.data, 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list, test_user_list = dataset['train_user_list'], dataset['test_user_list']
        train_pair = dataset['train_pair']
    print('Load complete')

    # Create dataset, model, optimizer
    dataset = TripletUniformPair(item_size, train_user_list, train_pair)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = BPR(user_size, item_size, args.dim).cuda()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    # Training
    smooth_loss = 0
    for _ in range(args.n_epochs):
        for idx, (u, i, j) in enumerate(loader):
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
                                                      train_user_list,
                                                      test_user_list,
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
                        default=os.path.join('preprocessed', 'ml-1m.pickle'),
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
                        default=20,
                        help="Period for printing smoothing loss during training")
    parser.add_argument('--eval_every',
                        type=int,
                        default=10000,
                        help="Period for evaluating precision and recall during training")
    parser.add_argument('--save_every',
                        type=int,
                        default=10000,
                        help="Period for saving model during training")
    parser.add_argument('--model',
                        type=str,
                        default=os.path.join('output', 'bpr.pt'),
                        help="File path for model")
    args = parser.parse_args()
    main(args)
