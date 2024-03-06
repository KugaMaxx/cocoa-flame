import numpy as np

import torch
import torch.nn as nn

from typing import Dict, Iterable
from collections import defaultdict


class Evaluator(object):
    def __init__(self, aet_ids: Dict, cat_ids: Dict) -> None:
        self.aet_ids = aet_ids
        self.cat_ids = cat_ids

        self.ious = defaultdict(list)
        self.stats = [] # 用于储存每次 batch 的总结，相当于 self.evalImgs

    def evaluate(self, outputs, targets):
        # 对逐个 aRng 和 maxDet 进行操作
        aet_id = self.aet_ids['name']
        cat_id = self.cat_ids['name']
        maxDet = 0
        aRng = 0

        if not (len(self.iou[aet_id, cat_id]) > 0):
            return []

        # 对 outputs 的结果根据 score 进行排序
        tgt_ind = np.argsort([], kind='stable')[:maxDet]  # 里面放入 score
        
        iou = self.iou[aet_id, cat_id][:, tgt_ind]

    def _calc_iou(self):
        pass

    def summarize(self):
        # 先将测评结果合并
        self.stat = np.concatenate(self.stat, 2)


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
    total_loss = 0
    for batch_idx, (samples, targets) in enumerate(data_loader):
        # set models and criterion to evaluate
        model.eval()
        criterion.eval()
        
        # inference
        outputs = model(samples)
        loss = criterion(outputs, targets)
        total_loss += loss.item()

    mean_loss = total_loss / (batch_idx + 1)

    return mean_loss
