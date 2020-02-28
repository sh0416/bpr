"""Utility functions for calculating evaluation metrics by accelerating gpu. Work In Progress"""
import torch

class VariableShapeLists(object):
    def __init__(self, lists):
        """
        Args:
            lists (list(torch.LongTensor)): list of tensors which have different shape.
        """
        super().__init__()
        # Initialize `batch_size`
        self.batch_size = len(lists)
        
        # Build up `shape`
        self.shapes = [x.shape for x in lists]
        
        # Build up `indice`
        self.indices = torch.empty((self.batch_size+1,), dtype=torch.long)
        self.indices[0], tmp = 0, 0
        for idx, list in enumerate(lists, start=1):
            tmp += list.numel()
            self.indices[idx] = tmp
        
        # Build up `data`
        self.data = torch.empty((self.indices[self.batch_size],))
        for idx, list in enumerate(lists, start=0):
            self.data[self.indice[idx]:self.indice[idx+1]] = list
            
    def get(self, idx):
        return self.data[self.indice[idx]:self.indice[idx+1]]
    
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


def calculate_correct_mask(pred, true):
    """Calculate correct mask.
    
    Args:
        pred (VariableShapeLists):
        true (VariableShapeLists):
        
    Returns:
        (VariableShapeLists, VariableShapeLists) the element in the list is torch.BooleanTensor
    """
    # Get `batch_size`
    assert pred.batch_size == true.batch_size, "pred and true have different batch size"
    pred
    # create mask matrix
    mask = torch.zeros(pred.batch_size, 
    

def accuracy(pred, true, total_size):
    """Calculate accuracy.
    
    Args:
        pred (VariableLengthList): 
        true (VariableLengthList): 
        total_size (torch.FloatTensor): [batch_size] or scalar tensor when the `total_size` is equal over all examples.
    Returns:
        torch.FloatTensor [batch_size]
    """
    correct_mask = 
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
    
