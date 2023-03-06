import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler, BatchSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from utils.torch import worker_init_fn
from params import NUM_WORKERS


class LenMatchBatchSampler(BatchSampler):
    """
    Custom PyTorch Sampler that generate batches of similar length.
    Helps speed up training.
    """
    def __iter__(self):
        buckets = [[]] * 1000
        yielded = 0

        for idx in self.sampler:
            bucket_id = self.sampler.data_source[idx][0].size(-1) // 20
            if len(buckets[bucket_id]) == 0:
                buckets[bucket_id] = []

            buckets[bucket_id].append(idx)

            if len(buckets[bucket_id]) == self.batch_size:
                batch = list(buckets[bucket_id])
                yield batch
                yielded += 1
                buckets[bucket_id] = []

        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch

        assert (
            len(self) == yielded
        ), f"Expected {len(self)}, but yielded {yielded} batches"


class OrderedDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        assert num_replicas is not None
        assert rank is not None
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(np.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[
            self.rank * self.num_samples: self.rank * self.num_samples
            + self.num_samples
        ]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


def custom_collate_fn(batch):
    y = torch.stack([b[1] for b in batch])
    y_aux = torch.stack([b[2] for b in batch])

    #     print([b[0].size(-1) for b in batch])
    max_w = np.max([b[0].size(2) for b in batch])
    max_h = np.max([b[0].size(1) for b in batch])

    size = (len(batch), batch[0][0].size(0), max_h, max_w)
    x = torch.zeros(size, device=y.device)

    for i, b in enumerate(batch):
        mid = (max_w - b[0].size(-1)) // 2
        x[i, :, : b[0].size(1), mid: mid + b[0].size(2)] = b[0]

    return x, y, y_aux


def define_loaders(
    train_dataset,
    val_dataset,
    batch_size=32,
    val_bs=32,
    use_len_sampler=False,
    distributed=False,
    world_size=0,
    local_rank=0,
):
    """
    Builds data loaders.
    TODO

    Args:
        train_dataset (BCCDataset): Dataset to train with.
        val_dataset (BCCDataset): Dataset to validate with.
        batch_size (int, optional): Training batch size. Defaults to 32.
        val_bs (int, optional): Validation batch size. Defaults to 32.
        use_len_sampler (bool, optional): Whether to use len sampler. Defaults to False.
        distributed (bool, optional): Whether training is distributed. Defaults to False.
        world_size (int, optional): World size. Defaults to 0.
        local_rank (int, optional): Local rank. Defaults to 0.

    Returns:
       DataLoader: Train loader.
       DataLoader: Val loader.
    """
    sampler, val_sampler = None, None
    if distributed:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
            seed=world_size + local_rank,
        )

        val_sampler = OrderedDistributedSampler(
            val_dataset, num_replicas=world_size, rank=local_rank
        )

    if use_len_sampler:
        len_sampler = LenMatchBatchSampler(
            RandomSampler(train_dataset),
            batch_size=batch_size,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=len_sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=None,
            persistent_workers=True,
        )

    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            drop_last=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=None,
            persistent_workers=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        sampler=val_sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=None,
        persistent_workers=True,
    )

    return train_loader, val_loader
