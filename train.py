import argparse
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from models import build_model
from datasets import build_dataloader
from utils.misc import set_seed


def parse_args():
    parser = argparse.ArgumentParser()

    # common
    parser.add_argument('--device', default='cuda', type=str)
    
    # training strategy
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--learning_rate', default=1e-5, type=float)

    # dataset
    parser.add_argument('--dataset_file', default='dv_fire')
    parser.add_argument('--dataset_path', default='/home/kuga/Workspace/aedat_to_dataset/', type=str)
    parser.add_argument('--num_workers', default=2, type=int)

    # model
    parser.add_argument('--model_name', default='point_mlp', type=str)

    ## point_mlp

    ## local grouper

    ## criterion

    # checkpoint
    parser.add_argument('--start_epoch', default=0, type=int)

    return parser.parse_args()


def train(model, data_loader, optimizer, criterion):
    total_loss = 0
    for batch_idx, (samples, targets) in enumerate(data_loader):
        # set models and criterion to train
        model.train()
        criterion.train()
        
        # inference
        outputs = model(samples)
        loss = criterion(outputs, targets)

        # back propagation
        optimizer.zero_grad()
        loss.backward()

        # update
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        total_loss += loss.item()

    mean_loss = total_loss / (batch_idx + 1)

    return mean_loss


@torch.no_grad()
def evaluate(model, data_loader, optimizer, criterion):
    pass
    # for batch_idx, (samples, targets) in enumerate(data_loader):
    #     # set models and criterion to evaluate
    #     model.eval()
    #     criterion.eval()
        
    #     # inference
    #     points = torch.tensor(samples['events'].numpy()).to(device)
    #     outputs = model(points)

    #     # post process
        

    #     # evaluate


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
    model.to(device)
    
    # build training strategy
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, last_epoch=args.start_epoch - 1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-3, last_epoch=args.start_epoch - 1)

    # create tensor board writer
    writer = SummaryWriter(log_dir='./', flush_secs=30)

    # Train model
    for epoch in range(args.start_epoch, args.epochs):
        print(f"Epoch(%d/%s) Learning Rate %s:" % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        
        # training
        train_result = train(model, data_loader=data_loader_train, 
                             optimizer=optimizer, criterion=criterion)
        
        # validation
        test_result  = evaluate(model, data_loader=data_loader_val, 
                                optimizer=optimizer, criterion=criterion)

        # update
        scheduler.step()
        writer.add_scalar('loss', train_result, epoch)

        print(train_result)
    
    writer.close()