import os
import torch
import random
import numpy as np

from pathlib import Path
from typing import Dict


def set_seed(seed=0):
    # set python hash seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # set numpy seed
    np.random.seed(seed)

    # set torch seed
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:  # faster, less reproducible
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def load_checkpoint(model, stat: Dict, cpkt_path: str):
    checkpoint = torch.load(cpkt_path)

    model.load_state_dict(checkpoint['model'])
    for key, value in checkpoint.items():
        if key == 'model': continue
        stat[key] = value

    return model, stat


def save_checkpoint(model, stat: Dict, cpkt_path: str):
    stat['model'] = model.state_dict()
    torch.save(stat, cpkt_path)
