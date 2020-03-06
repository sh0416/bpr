import torch
import unittest
from bpr.metrics import VariableShapeList, vsl_intersection


def test_create_variable_shape_list():
    zero_tensor = torch.zeros([2])
    one_tensor = torch.ones([1])
    variable_list = VariableShapeList.from_tensors([zero_tensor, one_tensor])
    assert variable_list.batch_size == 2
    assert torch.all(torch.eq(zero_tensor, variable_list[0])).item()


def test_calculate_intersection():
    pred1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    pred2 = torch.tensor([1, 3, 5, 7, 9, 11], dtype=torch.long)
    pred3 = torch.tensor([1, 3, 3, 4, 5], dtype=torch.long)
    pred_list = VariableShapeList.from_tensors([pred1, pred2, pred3])

    true1 = torch.tensor([3, 4, 5, 6, 7], dtype=torch.long)
    true2 = torch.tensor([2, 4, 6, 8, 10], dtype=torch.long)
    true3 = torch.tensor([3, 4, 5, 6, 7], dtype=torch.long)
    true_list = VariableShapeList.from_tensors([true1, true2, true3])

    inter1 = torch.tensor([3, 4, 5], dtype=torch.long)
    inter2 = torch.tensor([], dtype=torch.long)
    inter3 = torch.tensor([3, 3, 4, 5], dtype=torch.long)
    intersection_list = vsl_intersection(pred_list,
                                            true_list)
    assert torch.all(torch.eq(intersection_list[0], inter1))
    assert torch.all(torch.eq(intersection_list[1], inter2))
