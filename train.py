import os
import random
import pickle
import argparse
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torch.utils.tensorboard import SummaryWriter

from metrics import VariableShapeList, vsl_precision, vsl_recall


class TripletUniformPair(IterableDataset):
    def __init__(self, num_item, user_list, pair, shuffle, num_epochs):
        self.num_item = num_item
        self.user_list = user_list
        self.pair = pair
        self.shuffle = shuffle
        self.num_epochs = num_epochs

    def __iter__(self):
        worker_info = get_worker_info()
        # Shuffle per epoch
        self.example_size = self.num_epochs * len(self.pair)
        self.example_index_queue = deque([])
        self.seed = 0
        if worker_info is not None:
            self.start_list_index = worker_info.id
            self.num_workers = worker_info.num_workers
            self.index = worker_info.id
        else:
            self.start_list_index = None
            self.num_workers = 1
            self.index = 0
        return self

    def __next__(self):
        if self.index >= self.example_size:
            raise StopIteration
        # If `example_index_queue` is used up, replenish this list.
        while len(self.example_index_queue) == 0:
            index_list = list(range(len(self.pair)))
            if self.shuffle:
                random.Random(self.seed).shuffle(index_list)
                self.seed += 1
            if self.start_list_index is not None:
                index_list = index_list[self.start_list_index::self.num_workers]
                # Calculate next start index
                self.start_list_index = (self.start_list_index + (self.num_workers - (len(self.pair) % self.num_workers))) % self.num_workers
            self.example_index_queue.extend(index_list)
        result = self._example(self.example_index_queue.popleft())
        self.index += self.num_workers
        return result

    def _example(self, idx):
        u = self.pair[idx][0]
        i = self.pair[idx][1]
        j = np.random.randint(self.num_item)
        while j in self.user_list[u]:
            j = np.random.randint(self.num_item)
        return u, i, j


class BPR(nn.Module):
    def __init__(self, user_size, item_size, dim, weight_decay):
        super().__init__()
        self.W = nn.Parameter(torch.empty(user_size, dim))
        self.H = nn.Parameter(torch.empty(item_size, dim))
        nn.init.xavier_normal_(self.W.data)
        nn.init.xavier_normal_(self.H.data)
        self.weight_decay = weight_decay

    def forward(self, u, i, j):
        """Return loss value.
        
        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
            i(torch.LongTensor): tensor stored item indexes which is prefered by user. [batch_size,]
            j(torch.LongTensor): tensor stored item indexes which is not prefered by user. [batch_size,]
        
        Returns:
            torch.FloatTensor
        """
        u = self.W[u, :]
        i = self.H[i, :]
        j = self.H[j, :]
        x_ui = torch.mul(u, i).sum(dim=1)
        x_uj = torch.mul(u, j).sum(dim=1)
        x_uij = x_ui - x_uj
        log_prob = F.logsigmoid(x_uij).sum()
        regularization = self.weight_decay * (u.norm(dim=1).pow(2).sum() + i.norm(dim=1).pow(2).sum() + j.norm(dim=1).pow(2).sum())
        return -log_prob + regularization

    def recommend(self, u, k):
        """Return recommended item list given users.

        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
            k(int): recommend k items.

        Returns:
            pred(torch.LongTensor): recommended item list sorted by preference.
                                    [batch_size, k]
        """
        u = self.W[u, :]
        x_ui = torch.mm(u, self.H.t())
        pred = torch.argsort(x_ui, dim=1, descending=True)[:, :k]
        return pred


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
        mask = user_emb.new_ones([min([batch, user_emb.shape[0]-i]), item_emb.shape[0]], dtype=torch.float)
        for j in range(batch):
            if i+j >= user_emb.shape[0]:
                break
            mask[j].scatter_(dim=0, index=train_user_list[i+j].cuda(), value=torch.tensor(0.0).cuda())
        # Calculate prediction value
        cur_result = torch.mm(user_emb[i:i+min(batch, user_emb.shape[0]-i), :], item_emb.t())
        cur_result = torch.sigmoid(cur_result)
        assert not torch.any(torch.isnan(cur_result))
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
            test = set(test_user_list[i])
            pred = set(result[i, :k].numpy().tolist())
            val = len(test & pred)
            precision += val / max([min([k, len(test)]), 1])
            recall += val / max([len(test), 1])
        precisions.append(precision / user_emb.shape[0])
        recalls.append(recall / user_emb.shape[0])
    return precisions, recalls


def evaluate_model(model, train_user_list, test_user_list, k=5):
    precisions, recalls = [], []
    for u_start in range(0, len(train_user_list), 512):
        u_end = min([u_start+512, len(train_user_list)])
        pred = model.recommend(torch.arange(u_start, u_end).cuda(), k=k).cpu()
        train_true = VariableShapeList.from_tensors(train_user_list[u_start:u_end])
        test_true = VariableShapeList.from_tensors(test_user_list[u_start:u_end])
        #train_true = train_true.to(torch.device(torch.cuda.current_device()))
        train_pred = [pred[i] for i in range(pred.shape[0])]
        train_pred = VariableShapeList.from_tensors(train_pred)
        precisions.append(vsl_precision(train_pred, train_true))
        recalls.append(vsl_recall(train_pred, train_true))
    precisions = torch.cat(precisions, dim=0)
    recalls = torch.cat(recalls, dim=0)
    precision = precisions.mean()
    recall = recalls.mean()
    return precision, recall


def main(args):
    # Initialize seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load preprocess data
    with open(args.data, 'rb') as f:
        dataset = pickle.load(f)
        user_size = dataset['user_size']
        item_size = dataset['item_size']
        train_user_list = dataset['train_user_list']
        test_user_list = dataset['test_user_list']
        train_pair = dataset['train_pair']
    print('Load complete')

    # Convert list into tensor
    train_user_list = [torch.tensor(x, dtype=torch.long) for x in train_user_list]
    test_user_list = [torch.tensor(x, dtype=torch.long) for x in test_user_list]

    # Create dataset, model, optimizer
    dataset = TripletUniformPair(item_size, train_user_list, train_pair, True, args.n_epochs)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=16)
    model = BPR(user_size, item_size, args.dim, args.weight_decay).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter()

    import datetime
    start = datetime.datetime.now()
    evaluate_model(model, train_user_list, test_user_list)
    end = datetime.datetime.now()
    print('New elapsed time: %s' % (end - start))
    start = datetime.datetime.now()
    plist, rlist = precision_and_recall_k(model.W.detach(),
                                            model.H.detach(),
                                            train_user_list,
                                            test_user_list,
                                            klist=[1, 5, 10])
    end = datetime.datetime.now()
    print('Old elapsed time: %s' % (end - start))
    assert False
    # Training
    smooth_loss = 0
    idx = 0
    for u, i, j in loader:
        optimizer.zero_grad()
        loss = model(u, i, j)
        loss.backward()
        optimizer.step()
        writer.add_scalar('train/loss', loss, idx)
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
            p, r = evaluate_model(model, train_user_list, test_user_list)
            print('P@5: %.4f, R@5: %.4f' % (p, r))
            writer.add_scalars('eval', {'P@1': plist[0],
                                                    'P@5': plist[1],
                                                    'P@10': plist[2]}, idx)
            writer.add_scalars('eval', {'R@1': rlist[0],
                                                'R@5': rlist[1],
                                                'R@10': rlist[2]}, idx)
        if idx % args.save_every == (args.save_every - 1):
            dirname = os.path.dirname(os.path.abspath(args.model))
            os.makedirs(dirname, exist_ok=True)
            torch.save(model.state_dict(), args.model)
        idx += 1


if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                        type=str,
                        default=os.path.join('preprocessed', 'ml-1m.pickle'),
                        help="File path for data")
    # Seed
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="Seed (For reproducability)")
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
                        default=0.025,
                        help="Weight decay factor")
    # Training
    parser.add_argument('--n_epochs',
                        type=int,
                        default=500,
                        help="Number of epoch during training")
    parser.add_argument('--batch_size',
                        type=int,
                        default=4096,
                        help="Batch size in one iteration")
    parser.add_argument('--print_every',
                        type=int,
                        default=20,
                        help="Period for printing smoothing loss during training")
    parser.add_argument('--eval_every',
                        type=int,
                        default=1000,
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
