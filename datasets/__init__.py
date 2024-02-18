import torch
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler, BatchSampler

from .dv_fire.dv_fire import DvFire


__all__ = {
    'dv_fire': DvFire
}

def collate_fn(batch):
    return tuple(batch)


def build_dataloader(args, partition):
    dataset = __all__[args.dataset_file](
        dataset_path=args.dataset_path,
        partition=partition
    )

    if partition is 'train':
        sampler = RandomSampler(dataset)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    else:
        sampler = SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=False)
    
    data_loader = DataLoader(dataset=dataset, collate_fn=collate_fn, 
                             batch_sampler=batch_sampler, num_workers=args.num_workers)

    return data_loader
