import torch
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler, BatchSampler, DistributedSampler

from .dv_fire.dv_fire import DvFire


def collate_fn(batch):
    batch = list(zip(*batch))
    return tuple(batch)


def build_dataloader(args, partition):
    assert partition in ['test', 'train'], \
        "Support partitions are ['test', 'train']"

    candidate_datasets = {
        'dv_fire': DvFire
    }

    dataset = candidate_datasets[args.dataset_file](
        file_path=args.dataset_path,
        partition=partition
    )

    if partition == 'train':
        # TODO ddp
        # sampler = DistributedSampler(dataset) if args.distributed else RandomSampler(dataset)
        sampler = RandomSampler(dataset)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    elif partition == 'test':
        # sampler = SequentialSampler(dataset)
        # TODO ddp
        # sampler = DistributedSampler(dataset) if args.distributed else RandomSampler(dataset)
        sampler = RandomSampler(dataset)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=False)
    
    data_loader = DataLoader(dataset=dataset, collate_fn=collate_fn, 
                             batch_sampler=batch_sampler, num_workers=args.num_workers)

    return data_loader
