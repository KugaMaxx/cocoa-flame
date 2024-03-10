import os
import torch
import random
import logging
import numpy as np

from pathlib import Path
from datetime import datetime
from typing import Dict

from torch.utils.tensorboard import SummaryWriter


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


def create_logger(log_dir=None, log_level=logging.INFO):
    # create logger with level
    logger = logging.getLogger("Logger")
    logger.setLevel(log_level)

    if log_dir is not None:
        # check if directory exists
        path = Path(log_dir)
        path.mkdir(exist_ok=True)

        # set format
        formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')

        # create flle handler
        file_handler = logging.FileHandler(path / f"console.{datetime.now().isoformat()}.log")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def create_writer(log_dir=None):
    class EmptyWriter(object):
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    if log_dir is not None:
        # check if directory exists
        path = Path(log_dir)
        path.mkdir(exist_ok=True)

        # create tensor board writer
        writer = SummaryWriter(log_dir, flush_secs=30)
    else:
        writer = EmptyWriter(log_dir, flush_secs=30)
            
    return writer


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


# TOTAL_BAR_LENGTH = 65.
# last_time = time.time()
# begin_time = last_time
# def progress_bar(current, total, msg=None):
#     global last_time, begin_time
#     if current == 0:
#         begin_time = time.time()  # Reset for new bar.

#     cur_len = int(TOTAL_BAR_LENGTH*current/total)
#     rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

#     sys.stdout.write(' [')
#     for i in range(cur_len):
#         sys.stdout.write('=')
#     sys.stdout.write('>')
#     for i in range(rest_len):
#         sys.stdout.write('.')
#     sys.stdout.write(']')

#     cur_time = time.time()
#     step_time = cur_time - last_time
#     last_time = cur_time
#     tot_time = cur_time - begin_time

#     L = []
#     L.append('  Step: %s' % format_time(step_time))
#     L.append(' | Tot: %s' % format_time(tot_time))
#     if msg:
#         L.append(' | ' + msg)

#     msg = ''.join(L)
#     sys.stdout.write(msg)
#     # for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
#     #     sys.stdout.write(' ')

#     # Go back to the center of the bar.
#     # for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
#     #     sys.stdout.write('\b')
#     sys.stdout.write(' %d/%d ' % (current+1, total))

#     if current < total-1:
#         sys.stdout.write('\r')
#     else:
#         sys.stdout.write('\n')
#     sys.stdout.flush()


# def format_time(seconds):
#     days = int(seconds / 3600/24)
#     seconds = seconds - days*3600*24
#     hours = int(seconds / 3600)
#     seconds = seconds - hours*3600
#     minutes = int(seconds / 60)
#     seconds = seconds - minutes*60
#     secondsf = int(seconds)
#     seconds = seconds - secondsf
#     millis = int(seconds*1000)

#     f = ''
#     i = 1
#     if days > 0:
#         f += str(days) + 'D'
#         i += 1
#     if hours > 0 and i <= 2:
#         f += str(hours) + 'h'
#         i += 1
#     if minutes > 0 and i <= 2:
#         f += str(minutes) + 'm'
#         i += 1
#     if secondsf > 0 and i <= 2:
#         f += str(secondsf) + 's'
#         i += 1
#     if millis > 0 and i <= 2:
#         f += str(millis) + 'ms'
#         i += 1
#     if f == '':
#         f = '0ms'
#     return f
