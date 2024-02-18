import os
import torch
import random
import numpy as np


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
