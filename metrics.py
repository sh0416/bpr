"""Utility functions for calculating evaluation metrics by accelerating gpu."""
import pprint
import warnings
import torch
import vsl_cpp
#import vsl_cuda
from torch.utils.cpp_extension import load
vsl_cuda = load(name="vsl_cuda", sources=['vsl-extension/cuda/vsl_cuda.cpp', 'vsl-extension/cuda/vsl_cuda_kernel.cu'], verbose=True)


def shift_bit_length(x):
    return 1<<(x-1).bit_length()

class VariableShapeList(object):
    def __init__(self, indexes, data):
        super().__init__()
        assert indexes.device == data.device, \
                "`indexes` and `data` are stored in different storage"
        self.batch_size = indexes.shape[0] - 1
        self.indexes = indexes
        self.data = data

    @classmethod
    def from_tensors(cls, tensors):
        """
        Args:
            tensors (list(torch.LongTensor)): list of tensors which have different shape.
                                              For now, operation is only supported for one dimensional tensor.
        """
        assert len(tensors) > 0, "`tensors` is empty"
        assert all([len(x.shape) == 1 for x in tensors]), \
                "Some elements in `tensors` are not one dimensional"
        assert all([tensors[0].device==x.device for x in tensors]), \
                "Some elements in `tensors` are in different storage"
        # Initialize `batch_size`
        batch_size = len(tensors)
        
        # Build up `indexes`
        indexes = torch.empty((batch_size+1,), dtype=torch.long, device=tensors[0].device)
        indexes[0], tmp = 0, 0
        for idx, tensor in enumerate(tensors, start=1):
            tmp += tensor.numel()
            indexes[idx] = tmp
        
        # Build up `data`
        data = torch.zeros((shift_bit_length(indexes[batch_size].item()),), dtype=torch.long, device=tensors[0].device)
        for idx, tensor in enumerate(tensors, start=0):
            data[indexes[idx]:indexes[idx+1]] = tensor
        return cls(indexes, data)

    def __getitem__(self, idx):
        return self.data[self.indexes[idx]:self.indexes[idx+1]]
    
    def __str__(self):
        return str([self.__getitem__(i) for i in range(self.batch_size)])

    def get_size_tensor(self):
        return self.indexes[1:] - self.indexes[:-1]

    def to(self, device):
        self.data = self.data.to(device)
        self.indexes = self.indexes.to(device)
        return self

    @property
    def is_cuda(self):
        assert self.data.is_cuda == self.indexes.is_cuda
        return self.data.is_cuda


def vsl_intersection(list1, list2):
    """Calculate intersection VariableShapeList.
    For the first, we doesn't care about the duplicated item in the list.
    So, you must check whether the element in the list is not duplicated.
    
    Args:
        list1 (VariableShapeList): 
        list2 (VariableShapeList):
        
    Returns:
        VariableShapeList
    """
    # get `batch_size`
    assert list1.batch_size == list2.batch_size, \
            "`list1` and `list2` have different batch size"
    assert list1.is_cuda == list2.is_cuda, \
            "`list1` and `list2` are stored in different storage"
    if list1.is_cuda:
        data, indexes, mask, sum_ = vsl_cuda.vsl_intersection(list1.data,
                                                  list1.indexes,
                                                  list2.data,
                                                  list2.indexes)
        #print('MASK', mask)
        #print('SUM', sum_)
    else:
        data, indexes = vsl_cpp.vsl_intersection(list1.data,
                                                 list1.indexes,
                                                 list2.data,
                                                 list2.indexes)
    return VariableShapeList(indexes, data)
    

def vsl_precision(pred, true):
    """Calculate precision.
    
    Args:
        pred (VariableShapeList): 
        true (variableShapeList): 
    Returns:
        torch.FloatTensor [batch_size]
    """
    pred_size = pred.get_size_tensor().float()
    if not torch.all(pred.get_size_tensor()>0):
        warnings.warn("At least one of the denominator of precision "
                      "could be zero. Its nominator also must be zero,"
                      "so I substitute the denominator of precision to one.",
                      RuntimeWarning)
        pred_size = torch.where(pred_size>0, pred_size, torch.ones_like(pred_size))
    intersection = vsl_intersection(pred, true)
    intersection_size = intersection.get_size_tensor().float()
    return intersection_size / pred_size
    
    
def vsl_recall(pred, true):
    """Calculate recall.
    
    Args:
        pred (VariableShapeList): 
        true (variableShapeList): 
    Returns:
        torch.FloatTensor [batch_size]
    """
    true_size = true.get_size_tensor().float()
    if not torch.all(true_size>0):
        warnings.warn("At least one of the denominator of recall "
                      "could be zero. Its nominator also must be zero,"
                      "so I substitute the denominator of recall to one.",
                      RuntimeWarning)
        true_size = torch.where(true_size>0, true_size, torch.ones_like(true_size))
    intersection = vsl_intersection(pred, true)
    intersection_size = intersection.get_size_tensor().float()
    return intersection_size / true_size
    
    
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
            test = test_user_list[i]
            pred = set(result[i, :k].numpy().tolist())
            val = len(test & pred)
            precision += val / min([k, len(test)])
            recall += val / len(test)
        precisions.append(precision / user_emb.shape[0])
        recalls.append(recall / user_emb.shape[0])
    return precisions, recalls


def accuracy(pred, true, total_size):
    """Calculate accuracy.
    
    Args:
        pred (VariableLengthList): 
        true (VariableLengthList): 
        total_size (torch.FloatTensor): [batch_size] or scalar tensor when the `total_size` is equal over all examples.
    Returns:
        torch.FloatTensor [batch_size]
    """
    raise NotImplemented
    

def mean_average_precision(pred, true):
    raise NotImplemented
    
