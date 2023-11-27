import os
import logging
import datetime
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import Datasets
from models import Models, Criterions


def train(net: torch.nn.Module, data_loader: DataLoader, optimizer, criterion, device):
    net.train()
    with torch.set_grad_enabled(True):
        for batch_idx, (data, label) in enumerate(data_loader):
            optimizer.zero_grad()
            data, label = data.to(device), label.to(device)
            output = net(data)
            loss = criterion(output, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()


def validate(net: torch.nn.Module, data_loader: DataLoader, optimizer, criterion, device):
    net.eval()
    with torch.set_grad_enabled(False):
        for batch_idx, (data, label) in enumerate(data_loader):
            data, label = data.to(device), label.to(device)
            output = net(data)
            loss = criterion(output, label)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser("Wang Yiyi")
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='batch size in training')
    parser.add_argument("--device", type=str, default="cpu", 
                        help="select the device for inference.")
    parser.add_argument("--dataset", type=str, default="DvFire", 
                        help="select the dataset for inference.")
    parser.add_argument("--model", type=str, default="PointMLP", 
                        help="select the model for inference.")
    parser.add_argument("--seed", type=int, default=np.random.randint(0, 1000), 
                        help="if set a random seed. the result will be fully reproduced.")
    parser.add_argument('--epoch', type=int, default=300, 
                        help="number of epoch in training")
    parser.add_argument('--num_points', type=int, default=1024, 
                        help="Point Number.")
    parser.add_argument('--lr', type=float, default=0.1, 
                        help="learning rate in training.")
    parser.add_argument('--min_lr', type=float, default=0.005, 
                        help="min learning rate in training.")
    parser.add_argument('--weight_decay', type=float, default=2e-4, 
                        help="decay rate")
    args = parser.parse_args()

    # Set training seed
    if args.seed is not None:
        # 设置NumPy的随机种子
        np.random.seed(args.seed)
        
        # 设置PyTorch的随机种子
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.set_printoptions(10)

        # 关闭cudnn的benchmark选项，以获得确定性的结果
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # 设置Python的哈希种子，以确保在使用哈希的情况下也能获得可重复的结果
        os.environ['PYTHONHASHSEED'] = str(args.seed)

    # Load dataset
    train_loader = DataLoader(Datasets(name=args.dataset, partition="train"), 
                              num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader  = DataLoader(Datasets(name=args.dataset, partition="train"), 
                              num_workers=8, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Load inference model
    net = Models(args.model).to(args.device)
    if args.device == "cuda":
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark = True

    # Set criterion loss
    criterion = Criterions("entropy_loss")
        
    # Set training strategy
    optimizer = torch.optim.SGD(net.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.min_lr, last_epoch=-1)

    # Train model
    for epoch in range(args.epoch):
        print(f"Epoch(%d/%s) Learning Rate %s:" % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
        
        # training
        train_result = train(net, data_loader=train_loader, optimizer=optimizer, criterion=, device=args.device)
        
        # validation
        test_result  = validate()

        scheduler.step()