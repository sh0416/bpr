import torch
import pytest
from bpr.metrics import VariableShapeList, vsl_intersection, vsl_precision, vsl_recall


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


def test_calculate_intersection2():
    pred = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    true = torch.tensor([5, 1, 2], dtype=torch.long)
    pred_list = VariableShapeList.from_tensors([pred])
    true_list = VariableShapeList.from_tensors([true])
    intersection_list = vsl_intersection(pred_list,
                                            true_list)
    inter = torch.tensor([1, 2, 5], dtype=torch.long)
    print(intersection_list[0], intersection_list.data)
    assert torch.all(torch.eq(intersection_list[0], inter))



def test_calculate_precision_zero():
    pred = torch.tensor([6, 7, 8, 9, 10], dtype=torch.long)
    true = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    pred = VariableShapeList.from_tensors([pred])
    true = VariableShapeList.from_tensors([true])

    precision = vsl_precision(pred, true)
    assert precision[0].item() == 0.0


def test_calculate_precision_one():
    pred = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    true = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    pred = VariableShapeList.from_tensors([pred])
    true = VariableShapeList.from_tensors([true])

    precision = vsl_precision(pred, true)
    assert precision[0].item() == 1.0


def test_calculate_precision_float():
    pred = torch.tensor([1, 2, 3, 5, 7, 6], dtype=torch.long)
    true = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    pred = VariableShapeList.from_tensors([pred])
    true = VariableShapeList.from_tensors([true])

    precision = vsl_precision(pred, true)
    assert torch.isclose(precision[0], torch.tensor(4/6))


def test_calculate_precision_multiple():
    pred1 = torch.tensor([6, 7, 8, 9, 10], dtype=torch.long)
    pred2 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    pred3 = torch.tensor([1, 2, 3, 5, 7, 6], dtype=torch.long)
    true1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    true2 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    true3 = torch.tensor([1, 2, 3, 5], dtype=torch.long)
    pred = VariableShapeList.from_tensors([pred1, pred2, pred3])
    true = VariableShapeList.from_tensors([true1, true2, true3])

    precision = vsl_precision(pred, true)
    assert torch.allclose(precision, torch.tensor([0, 1, 4/6], dtype=torch.float))


def test_calculate_precision_edge():
    pred = torch.tensor([], dtype=torch.long)
    true = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    pred = VariableShapeList.from_tensors([pred])
    true = VariableShapeList.from_tensors([true])

    with pytest.raises(ZeroDivisionError):
        precision = vsl_precision(pred, true)


def test_calculate_precision_edge_multiple():
    pred1 = torch.tensor([6, 7, 8, 9, 10], dtype=torch.long)
    pred2 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    pred3 = torch.tensor([1, 2, 3, 5, 7, 6], dtype=torch.long)
    pred4 = torch.tensor([], dtype=torch.long)
    true1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    true2 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    true3 = torch.tensor([1, 2, 3, 5], dtype=torch.long)
    true4 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    pred = VariableShapeList.from_tensors([pred1, pred2, pred3, pred4])
    true = VariableShapeList.from_tensors([true1, true2, true3, true4])

    with pytest.raises(ZeroDivisionError):
        precision = vsl_precision(pred, true)


def test_calculate_recall_zero():
    pred = torch.tensor([6, 7, 8, 9, 10], dtype=torch.long)
    true = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    pred = VariableShapeList.from_tensors([pred])
    true = VariableShapeList.from_tensors([true])

    recall = vsl_recall(pred, true)
    assert recall[0].item() == 0.0


def test_calculate_recall_one():
    pred = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    true = torch.tensor([5, 1, 2], dtype=torch.long)
    pred = VariableShapeList.from_tensors([pred])
    true = VariableShapeList.from_tensors([true])

    recall = vsl_recall(pred, true)
    assert recall[0].item() == 1.0


def test_calculate_recall_float():
    pred = torch.tensor([1, 2, 3, 5, 7, 6], dtype=torch.long)
    true = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    pred = VariableShapeList.from_tensors([pred])
    true = VariableShapeList.from_tensors([true])

    recall = vsl_recall(pred, true)
    assert torch.isclose(recall[0], torch.tensor(4/5))


def test_calculate_recall_multiple():
    pred1 = torch.tensor([6, 7, 8, 9, 10], dtype=torch.long)
    pred2 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    pred3 = torch.tensor([1, 2, 3, 5, 7, 6], dtype=torch.long)
    true1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    true2 = torch.tensor([5, 1, 2], dtype=torch.long)
    true3 = torch.tensor([1, 2, 3, 5, 4], dtype=torch.long)
    pred = VariableShapeList.from_tensors([pred1, pred2, pred3])
    true = VariableShapeList.from_tensors([true1, true2, true3])

    recall = vsl_recall(pred, true)
    print(recall)
    assert torch.allclose(recall, torch.tensor([0., 1., 0.8], dtype=torch.float32))


def test_calculate_recall_edge():
    pred = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    true = torch.tensor([], dtype=torch.long)
    pred = VariableShapeList.from_tensors([pred])
    true = VariableShapeList.from_tensors([true])

    with pytest.raises(ZeroDivisionError):
        recall = vsl_recall(pred, true)


def test_calculate_recall_edge_multiple():
    pred1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    pred2 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    pred3 = torch.tensor([1, 2, 3, 5], dtype=torch.long)
    pred4 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    true1 = torch.tensor([6, 7, 8, 9, 10], dtype=torch.long)
    true2 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    true3 = torch.tensor([1, 2, 3, 5, 7, 6], dtype=torch.long)
    true4 = torch.tensor([], dtype=torch.long)
    pred = VariableShapeList.from_tensors([pred1, pred2, pred3, pred4])
    true = VariableShapeList.from_tensors([true1, true2, true3, true4])

    with pytest.raises(ZeroDivisionError):
        recall = vsl_recall(pred, true)
