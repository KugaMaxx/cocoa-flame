import argparse
import random

import numpy as np

import torch
from torch.utils.data import DataLoader

from models import build_model
from datasets import build_dataloader
from utils.misc import set_seed


def parse_args():
    parser = argparse.ArgumentParser()

    # common
    parser.add_argument('--device', default='cuda', type=str)
    
    # training strategy
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=2e-4, type=float)

    # dataset
    parser.add_argument('--dataset_file', default='dv_fire')
    parser.add_argument('--dataset_path', default='/home/kuga/Workspace/aedat_to_dataset', type=str)
    parser.add_argument('--num_workers', default=111, type=int)

    # model
    ## point_mlp

    ## local grouper


    # criterion

    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # fix the seed for reproducibility
    set_seed(args.seed)

    # build dataset
    data_loader_train = build_dataloader(args, partition='train') 
    data_loader_val   = build_dataloader(args, partition='test')

    for data, resolution, target in data_loader_train:
        breakpoint()

    # # build model
    # device = torch.device(args.device)

    # model, criterion = build_model(args)
    # model.to(args.device)
    
    # # build training strategy


    # # start training
    # for epoch in range(args.epoch):
    #     # training
    #     train_result = train(model, data_loader=data_loader_train, optimizer=optimizer, 
    #                          criterion=criterion, device=args.device)
        
    #     # validation
    #     test_result  = evaluate(model, data_loader=data_loader_val, optimizer=optimizer,
    #                             criterion=criterion, device=args.device)

    #     scheduler.step()

    #     print(test_result["acc"])
