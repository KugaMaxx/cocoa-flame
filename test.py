import argparse
from pathlib import Path

import torch

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
    parser.add_argument('--dataset_file', default='flade')
    parser.add_argument('--dataset_path', default='/data/Ding/FlaDE/', type=str)
    parser.add_argument('--num_workers', default=1, type=int)

    # model
    parser.add_argument('--model_name', default='point_mlp', type=str)

    ## point_mlp

    ## local grouper

    ## criterion

    # checkpoint
    parser.add_argument('--resume', action='store_false')
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
    writer = create_writer(args.log_dir)

    # create logger
    logger = create_logger(args.log_dir)

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
        model, stat = load_checkpoint(model, stat, checkpoint_dir / "0514_best_checkpoint.pth")

    # set training strategy
    optimizer = torch.optim.AdamW(model.parameters(), lr=stat['learning_rate'], weight_decay=stat['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, last_epoch=stat['epoch'] - 1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - stat['epoch'], eta_min=1e-3, last_epoch=stat['epoch'] - 1)
    
    # validation
    test_result = evaluate(model, criterion=criterion, data_loader=data_loader_val)

    # writer.close()