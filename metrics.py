"""Utility functions for calculating evaluation metrics by accelerating gpu. Work In Progress"""
import torch

class DifferentLengthList(object):
    def __init__(self, lists):
        """
        Args:
            lists (list(torch.LongTensor)): list of tensor which has different length. 
                                            For now, we limit the element in the list is one-dimensional tensor.
        """
        # Initialize `batch_size`
        self.batch_size = len(lists)
        
        # Build up `indice`
        self.indice = torch.empty((self.batch_size+1,), dtype=torch.long)
        self.indice[0], tmp = 0, 0
        for idx, list in enumerate(lists, start=1):
            tmp += list.shape[0]
            self.indice[idx] = tmp
        
        # Build up `data`
        self.data = torch.empty((self.indice[self.batch_size],))
        for idx, list in enumerate(lists, start=0):
            self.data[self.indice[idx]:self.indice[idx+1]] = list
            
    def get(self, idx):
        return self.data[self.indice[idx]:self.indice[idx+1]]
    
    
def accuracy(pred, true, total_size):
    """Calculate accuracy.
    
    Args:
        pred (torch.LongTensor): [batch_size, prediction_length]
        true (BatchSentence): [batch_size, true_length]
        total_size (torch.FloatTensor): [batch_size] or scalar tensor when the `total_size` is equal over all examples.
    Returns:
        torch.FloatTensor [batch_size]
    """
    raise NotImplemented
    

def precision(pred, true):
    """Calculate precision.
    
    Args:
        pred (torch.LongTensor): [batch_size, prediction_length]
        true (torch.LongTensor): [batch_size, true_length]
    Returns:
        torch.FloatTensor [batch_size]
    """
    raise NotImplemented
    
    
def recall(pred, true):
    raise NotImplemented
    
    
def mean_average_precision(pred, true):
    raise NotImplemented
    
