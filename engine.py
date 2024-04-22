import torch
import torch.nn as nn

from typing import Dict, Iterable

from utils.misc import load_logger
from utils.eval import Evaluator
from utils.plot import plot_detection_result, plot_projected_events, plot_rescaled_image


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
    # criterion.temperature = criterion.temperature - 0.0002

    return {
        'mean_loss': total_loss / (batch_idx + 1),
    }


@torch.no_grad()
def evaluate(model: nn.Module, criterion: nn.Module, data_loader: Iterable):
    # eval = Evaluator(aet_ids=data_loader.dataset.aet_ids, 
    #                  cat_ids=data_loader.dataset.cat_ids)
    eval = Evaluator(aet_ids=data_loader.dataset.aet_ids, 
                     cat_ids={'fire': 0})

    total_loss = 0
    for batch_idx, (samples, targets) in enumerate(data_loader):
        # set models and criterion to evaluate
        model.eval()
        criterion.eval()
        
        # inference
        outputs = model(samples)
        loss = criterion(outputs, targets)

        # one batch step
        total_loss += loss.item()

        # keep only predictions with greater than 0.7 confidence
        for i, _ in enumerate(outputs):
            idn = torch.logical_and(outputs[i]['scores'] > 0.7, outputs[i]['labels'] == 0)
            outputs[i]['scores'] = outputs[i]['scores'][idn]
            outputs[i]['labels'] = outputs[i]['labels'][idn]
            outputs[i]['bboxes'] = outputs[i]['bboxes'][idn]

        eval.update(outputs, targets)

        # # TODO ready to delete
        # def plot(sample, target, output, i):
        #     import cv2
        #     import numpy as np
        #     image  = np.zeros((260, 346)) if sample['frames'] is None else sample['frames'].numpy()
        #     events = np.zeros((1, 4))     if sample['events'] is None else sample['events'].numpy()

        #     image = plot_projected_events(image, events)
        #     image = plot_rescaled_image(image)
        #     image = plot_detection_result(image, 
        #                                   bboxes=(target['bboxes']).tolist(),
        #                                   labels=(target['labels']).tolist(),
        #                                   colors=[(0, 0, 255)])
            
        #     image = plot_detection_result(image, 
        #                                   bboxes=output['bboxes'],
        #                                   labels=output['labels'],
        #                                   scores=output['scores'],
        #                                   colors=[(255, 0, 0)])
        #     cv2.imwrite(f'./result_{i}.png', image)
        #     return True

        # if batch_idx == 0:
        #     results = [plot(sample, target, output, i) \
        #                for i, (sample, target, output) in enumerate(zip(samples, targets, outputs))]
        #     eval.summarize()
        #     break
        # # TODO ready to delete

    # evaluation
    stats = eval.summarize()
    logger = load_logger()
    logger.info('\n' + '\n'.join(f'{info}: {value:.3f}' for info, value in stats))

    return {
        'mean_loss': total_loss / (batch_idx + 1),
        'mAP': stats[0][1], 
        'mAP_50': stats[1][1]
    }
