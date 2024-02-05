import argparse
import random

import numpy as np
import torch

from models import build_model
from datasets 


def parse_args():
    parser = argparse.ArgumentParser()

    # Training strategy
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=2e-4, type=float)

    # Dataset
    parser.add_argument('--dataset_file', default='/home/kuga/Workspace/aedat_to_dataset', type=str)
    parser.add_argument('--num_workers', default=2, type=int)

    # Model
    ##
    parser.add_argument('--k_neighbors', default=[32, 32, 32, 32], type=int)
    parser.add_argument()

    ## Local Grouper
    parser.add_argument('--k_neighbors', default=[32, 32, 32, 32], type=int)
    parser.add_argument()

    # Criterion

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # fix the seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = build_model(args)
    model.to(args.device)

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
