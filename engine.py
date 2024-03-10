import torch
import torch.nn as nn
from torchvision.ops import box_iou

from typing import Dict, Iterable

from utils.eval import Evaluator


def train(model: nn.Module, criterion: nn.Module, data_loader: Iterable, 
          optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler):
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

        # one batch step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        total_loss += loss.item()

    # one epoch step
    scheduler.step()
    mean_loss = total_loss / (batch_idx + 1)

    return mean_loss


@torch.no_grad()
def evaluate(model: nn.Module, criterion: nn.Module, data_loader: Iterable):
    eval = Evaluator(aet_ids=data_loader.dataset.aet_ids, 
                     cat_ids=data_loader.dataset.cat_ids)

    total_loss = 0
    for batch_idx, (samples, targets) in enumerate(data_loader):
        # set models and criterion to evaluate
        model.eval()
        criterion.eval()
        
        # inference
        outputs = model(samples)
        loss = criterion(outputs, targets)
        total_loss += loss.item()

        eval.update(outputs, targets)

    mean_loss = total_loss / (batch_idx + 1)

    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    return mean_loss
    