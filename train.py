import argparse
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from models import build_model
from datasets import build_dataloader
from utils.misc import set_seed, create_logger, create_writer, save_checkpoint, load_checkpoint
from engine import train, evaluate


def parse_args():
    parser = argparse.ArgumentParser()

    # common
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--output_dir', default='./checkpoint', type=str)
    
    # training strategy
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--learning_rate', default=1e-5, type=float)

    # dataset
    parser.add_argument('--dataset_file', default='dv_fire')
    parser.add_argument('--dataset_path', default='./datasets/dv_fire/aedat_to_data/', type=str)
    parser.add_argument('--num_workers', default=1, type=int)

    # model
    parser.add_argument('--model_name', default='point_mlp', type=str)

    ## point_mlp

    ## local grouper

    ## criterion

    # checkpoint
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint_dir', default='./checkpoint', type=str)
    # parser.add_argument('--cpkt_name', default='last_checkpoint', type=str)
    parser.add_argument('--checkpoint_epoch', default=50, type=int)

    # logging
    parser.add_argument('--log_dir', default=None, type=str)

    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # fix for reproducibility
    seed = set_seed(args.seed)

    # create tensor board writer
    writer = create_writer(args.log_dir) if args.log_dir else None

    # create logger
    logger = create_logger(args.log_dir) if args.log_dir else None

    # initialize
    stat = dict(
        epoch = 0, args = args,
        weight_decay  = args.weight_decay,
        learning_rate = args.learning_rate,
    )

    # build dataset
    data_loader_train = build_dataloader(args, partition='train') 
    data_loader_val   = build_dataloader(args, partition='test')

    # build model
    device = torch.device(args.device)
    model, criterion = build_model(args)
    model.to(device)
    
    # load checkpoint
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        model, stat = load_checkpoint(model, stat, checkpoint_dir / "last_checkpoint.pth")

    # set training strategy
    optimizer = torch.optim.AdamW(model.parameters(), lr=stat['learning_rate'], weight_decay=stat['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, last_epoch=stat['epoch'] - 1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - stat['epoch'], eta_min=1e-3, last_epoch=stat['epoch'] - 1)
    
    # train model
    for epoch in range(stat['epoch'], args.epochs):
        logger.info(f"Epoch(%d/%s) Learning Rate %s:" % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        
        # training
        train_result = train(model, criterion=criterion, data_loader=data_loader_train, optimizer=optimizer, scheduler=scheduler)
        
        # validation
        test_result  = evaluate(model, criterion=criterion, data_loader=data_loader_val)

        # update
        if args.log_dir:
            writer.add_scalar('loss', train_result, epoch)

        # if (epoch + 1) % args.checkpoint_epoch == 0:
        #     save_checkpoint(model, stat, checkpoint_dir / "last_checkpoint.pth")

        print(train_result)
    
    # ending
    if args.log_file:
        writer.close()
