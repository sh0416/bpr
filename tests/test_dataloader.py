from collections import Counter
from torch.utils.data import DataLoader
from bpr.data import DatasetPipeline


def test_create_dataset():
    dataset = list(range(100))
    pipeline = DatasetPipeline(1, 1, False, dataset)
    for i, j in zip(dataset, pipeline):
        assert i == j[0]


def test_dataset_shuffle():
    dataset = list(range(100))
    pipeline = DatasetPipeline(1, 1, True, dataset)
    produced_dataset = []
    for i in pipeline:
        produced_dataset.append(i)
    produced_dataset_stats = Counter([x for batch in produced_dataset for x in batch])
    assert set(produced_dataset_stats.keys()) == set(dataset)
    assert all([x == 1 for x in produced_dataset_stats.values()])
    

def test_dataset_epoch():
    dataset = list(range(100))
    pipeline = DatasetPipeline(5, 1, False, dataset)
    produced_dataset = []
    for i in pipeline:
        produced_dataset.append(i)
    dataset = [x for _ in range(5) for x in dataset]
    assert all([x[0] == y for x, y in zip(produced_dataset, dataset)])


def test_dataset_batch():
    dataset = list(range(100))
    pipeline = DatasetPipeline(1, 7, False, dataset)
    true_examples = [dataset[i:i+7] for i in range(0, 100, 7)]

    produced_examples = []
    for i in pipeline:
        produced_examples.append(i)

    assert len(produced_examples) == len(true_examples)
    for prod, true in zip(produced_examples, true_examples):
        assert len(prod) == len(true)
        assert all([x == y for x, y in zip(prod, true)])
    

def test_dataset_epoch_shuffle():
    dataset = list(range(100))
    pipeline = DatasetPipeline(5, 1, True, dataset)
    produced_dataset = []
    for i in pipeline:
        produced_dataset.append(i)
    assert len(produced_dataset) == 5 * len(dataset)
    produced_dataset = [produced_dataset[i:i+len(dataset)] for i in range(0, len(produced_dataset), len(dataset))]
    for i in range(5):
        produced_dataset_stats = Counter([x for batch in produced_dataset[i] for x in batch])
        assert set(produced_dataset_stats.keys()) == set(dataset)
        assert all([x == 1 for x in produced_dataset_stats.values()])
    

def test_dataset_epoch_shuffle_batch():
    dataset = list(range(100))
    pipeline = DatasetPipeline(5, 7, True, dataset)

    produced_examples = []
    for i in pipeline:
        produced_examples.append(i)

    # Shuffle & epoch test
    stats = Counter()
    for i in produced_examples:
        stats.update(i)
    assert set(stats.keys()) == set(dataset)
    assert all([x == 5 for x in stats.values()])

    # Batch test
    batch_examples = [dataset[i:i+7] for i in range(0, 100, 7)] * 5
    for prod, true in zip(produced_examples, batch_examples):
        assert len(prod) == len(true)


def test_use_dataloader_for_dataset_multiprocess():
    dataset = list(range(100))
    pipeline = DatasetPipeline(5, 7, True, dataset)
    loader_single = DataLoader(pipeline, batch_size=None, batch_sampler=None, num_workers=0)
    loader_multi = DataLoader(pipeline, batch_size=None, batch_sampler=None, num_workers=4)
    for idx, (i, j) in enumerate(zip(loader_single, loader_multi)):
        print(idx)
        assert all([x == y for x, y in zip(i, j)])