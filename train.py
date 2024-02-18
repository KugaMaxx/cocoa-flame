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
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=2e-4, type=float)
    parser.add_argument('--learning_rate', default=0.1, type=float)
    parser.add_argument('--epochs', default=300, type=int)

    # dataset
    parser.add_argument('--dataset_file', default='dv_fire')
    parser.add_argument('--dataset_path', default='./datasets/dv_fire/aedat_to_data', type=str)
    parser.add_argument('--num_workers', default=2, type=int)

    # model
    ## point_mlp

    ## local grouper

    ## criterion

    # checkpoint
    parser.add_argument('--start_epoch', default=0, type=int)

    return parser.parse_args()


def train(model, data_loader, optimizer, criterion, device):
    for batch_idx, (samples, targets) in enumerate(data_loader):
        # set models and criterion to train
        model.train()
        criterion.train()
        
        # inference
        points = torch.tensor(samples['events'].numpy()).to(device)
        outputs = model(points)
        loss = criterion(outputs, targets)

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, data_loader, optimizer, criterion, device):
    for batch_idx, (samples, targets) in enumerate(data_loader):
        # set models and criterion to evaluate
        model.eval()
        criterion.eval()
        
        # inference
        points = torch.tensor(samples['events'].numpy()).to(device)
        outputs = model(points)

        # post process
        

        # evaluate



if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # fix the seed for reproducibility
    set_seed(args.seed)

    # build dataset
    data_loader_train = build_dataloader(args, partition='train') 
    data_loader_val   = build_dataloader(args, partition='test')

    # build model
    device = torch.device(args.device)

    model, criterion = build_model(args)
    model.to(args.device)
    
    # build training strategy
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=1e-3, last_epoch=args.start_epoch - 1)


    # Train model
    for epoch in range(args.start_epoch, args.epochs):
        print(f"Epoch(%d/%s) Learning Rate %s:" % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
        
        # training
        train_result = train(model, data_loader=data_loader_train, optimizer=optimizer, 
                             criterion=criterion, device=args.device)
        
        # validation
        test_result  = evaluate(model, data_loader=data_loader_val, optimizer=optimizer,
                                criterion=criterion, device=args.device)

        scheduler.step()
