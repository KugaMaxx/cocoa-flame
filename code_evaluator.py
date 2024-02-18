import argparse
import random

import numpy as np

import torch
from torch.utils.data import DataLoader

from models import build_model
from datasets import build_dataloader
from utils.misc import set_seed


class TODOEvaluator():
    def __init__(self) -> None:
        pass


if __name__ == '__main__':
    # test hungarian matcher
    num_labels, num_queries, num_classes = [2, 4], 100, 92
    
    targets = list()
    for i, num_label in enumerate(num_labels):
        targets.append({
            'labels': torch.randint(0, num_classes, [num_label]),
            'boxes': torch.rand([num_label, 4]),
            'resolution': (346, 260)
        })
    
    batch_size = len(num_labels)
    outputs = {
        "pred_logits": torch.rand([batch_size, num_queries, num_classes]).softmax(-1),
        "pred_boxes": torch.rand([batch_size, num_queries, 4])
    }

    # TODO: add an evaluator
    evaluator = TODOEvaluator(outputs, targets, 'bbox')
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()