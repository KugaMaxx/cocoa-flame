import os
import sys
import time
import logging
import datetime
import argparse
import numpy as np
import sklearn.metrics as metrics

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import Datasets
from models import Models, Criterions

from tmp import ModelNet40

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    # for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
    #     sys.stdout.write(' ')

    # Go back to the center of the bar.
    # for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
    #     sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def train(net: torch.nn.Module, data_loader: DataLoader, optimizer, criterion, device):
    # --- add ---
    train_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    time_cost = datetime.datetime.now()
    # --- add ---
    
    net.train()
    with torch.set_grad_enabled(True):
        for batch_idx, (data, label) in enumerate(data_loader):
            optimizer.zero_grad()
            data, label = data.to(device), label.to(device).squeeze()
            # --- add ---
            data = data.permute(0, 2, 1)  # so, the input data shape is [batch, 3, 1024]
            # --- add ---
            pred = net(data)
            loss = criterion(pred, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()
            
            # --- add ---
            train_loss += loss.item()
            preds = pred.max(dim=1)[1]

            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

            total += label.size(0)
            correct += preds.eq(label).sum().item()

            progress_bar(batch_idx, len(data_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            # --- add ---
    
    # --- add ---
    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(train_true, train_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(train_true, train_pred))),
        "time": time_cost
    }
    # --- add ---


def validate(net: torch.nn.Module, data_loader: DataLoader, optimizer, criterion, device):
    # --- add ---
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    # --- add ---
    
    net.eval()
    with torch.set_grad_enabled(False):
        for batch_idx, (data, label) in enumerate(data_loader):
            data, label = data.to(device), label.to(device).squeeze()
            # --- add ---
            data = data.permute(0, 2, 1)  # so, the input data shape is [batch, 3, 1024]
            # --- add ---
            pred = net(data)
            loss = criterion(pred, label)

            # --- add ---
            test_loss += loss.item()
            preds = pred.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()

            progress_bar(batch_idx, len(data_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            # --- add ---

    # --- add ---
    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }
    # --- add ---


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser("Wang Yiyi")
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
    parser.add_argument("--device", type=str, default="cuda", help="select the device for inference.")
    parser.add_argument("--dataset", type=str, default="DvFire", help="select the dataset for inference.")
    parser.add_argument("--model", type=str, default="PointMLPElite", help="select the model for inference.")
    parser.add_argument("--seed", type=int, default=6, help="if set a random seed. the result will be fully reproduced.")
    parser.add_argument('--epoch', type=int, default=300, help="number of epoch in training")
    parser.add_argument('--num_points', type=int, default=1024, help="Point Number.")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate in training.")
    parser.add_argument('--min_lr', type=float, default=0.005, help="min learning rate in training.")
    parser.add_argument('--weight_decay', type=float, default=2e-4, help="decay rate")
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    # Set training seed
    if args.seed is not None:
        # set numpy seed
        np.random.seed(args.seed)
        
        # set pytorch seed
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.set_printoptions(10)

        # 关闭cudnn的benchmark选项，以获得确定性的结果
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # set python hash seed
        os.environ['PYTHONHASHSEED'] = str(args.seed)

    # Load dataset
    # train_loader = DataLoader(Datasets(name=args.dataset, partition="train"), 
    #                           num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # test_loader  = DataLoader(Datasets(name=args.dataset, partition="train"), 
    #                           num_workers=8, batch_size=args.batch_size, shuffle=False, drop_last=False)
    # --- add ---
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.batch_size // 2, shuffle=False, drop_last=False)
    # --- add ---

    # Load inference model
    net = Models(args.model).to(args.device)
    if args.device == "cuda":
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark = True

    # Set criterion loss
    criterion = Criterions(name="entropy_loss")
        
    # Set training strategy
    optimizer = torch.optim.SGD(net.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.min_lr, last_epoch=-1)

    # Train model
    for epoch in range(args.epoch):
        print(f"Epoch(%d/%s) Learning Rate %s:" % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
        
        # training
        train_result = train(net, data_loader=train_loader, optimizer=optimizer, 
                             criterion=criterion, device=args.device)
        
        # validation
        test_result  = validate(net, data_loader=test_loader, optimizer=optimizer,
                                criterion=criterion, device=args.device)

        scheduler.step()

        print(test_result["acc"])
