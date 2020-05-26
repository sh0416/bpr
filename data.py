import random
from collections import deque
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

class DatasetPipeline(IterableDataset):
    def __init__(self, num_epochs, batch_size, shuffle, dataset):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dataset = dataset

    def __iter__(self):
        worker_info = get_worker_info()
        # Shuffle per epoch
        self.example_size = self.num_epochs * ((len(self.dataset) + self.batch_size - 1) // self.batch_size)
        self.example_index_queue = deque([])
        self.seed = 0
        if worker_info is not None:
            self.worker_id = worker_info.id
            self.start_list_index = worker_info.id
            self.num_workers = worker_info.num_workers
            self.index = worker_info.id
        else:
            self.worker_id = None
            self.start_list_index = None
            self.num_workers = 1
            self.index = 0
        return self

    def __next__(self):
        if self.index >= self.example_size:
            raise StopIteration
        # If `example_index_queue` is used up, replenish this list.
        while len(self.example_index_queue) == 0:
            index_list = np.arange(len(self.dataset))
            # Shuffle if needed
            if self.shuffle:
                np.random.seed(self.seed)
                np.random.shuffle(index_list)
                self.seed += 1
            # Batching `index_list`
            index_list = [
                index_list[i:min(index_list.shape[0], i+self.batch_size)]
                for i in range(0, index_list.shape[0], self.batch_size)
            ]
            # Multiprocessing support
            if self.start_list_index is not None:
                len_epoch = len(index_list)
                index_list = index_list[self.start_list_index::self.num_workers]
                # Calculate next start index
                self.start_list_index = (self.start_list_index + (self.num_workers - (len_epoch % self.num_workers))) % self.num_workers
            self.example_index_queue.extend(index_list)
        result = self.dataset[self.example_index_queue.popleft()]
        self.index += self.num_workers
        return result
