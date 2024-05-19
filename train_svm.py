import math
import torch
import argparse
import numpy as np
from sklearn.svm import SVC

from utils.misc import set_seed
from utils.eval import Evaluator
from utils.plot import plot_projected_events
from datasets import build_dataloader

import dv_processing as dv
import dv_toolkit as kit

from models.scout import flame_scout

import alphashape
from shapely.geometry import Polygon


def parse_args():
    parser = argparse.ArgumentParser()
    
    # training strategy
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--seed', default=42, type=int)

    # dataset
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--dataset_file', default='flade')
    parser.add_argument('--dataset_path', default='/data/Ding/FlaDE/', type=str)
    return parser.parse_args()


def detect(sample, target):
    model = flame_scout.init(target['resolution'], candidate_num=1)
    model.accept(sample['events'])
    bboxes = model.detect()
    return bboxes


def extract(sample, target, bboxes):
    box_x = bboxes[0][0] * target['resolution'][0]
    box_y = bboxes[0][1] * target['resolution'][1]
    box_w = bboxes[0][2] * target['resolution'][0]
    box_h = bboxes[0][3] * target['resolution'][1]
    
    idn = np.logical_and(
        np.logical_and(sample['events'][:, 1] >= box_x, sample['events'][:, 1] <= (box_x + box_w)),
        np.logical_and(sample['events'][:, 2] >= box_y, sample['events'][:, 2] <= (box_y + box_h))
    ).bool()
    
    sample['events'] = sample['events'][idn]

    # # find alpha shape
    # import alphashape
    # from shapely.geometry import Polygon
    # shaper = alphashape.alphashape(sample['events'][..., 1:3], alpha=.0)
    # polygon = Polygon([(x, y) for x, y in zip(*shaper.exterior.xy)])

    feat = [
        len(sample['events']),  # 事件输出率
        box_w / box_h,          # 长宽比
        # polygon.area / (box_w * box_h), # 矩形度
        # 4 * math.pi * polygon.area / polygon.length ** 2, # 圆形度
    ]

    return feat


if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # fix for reproducibility
    seed = set_seed(args.seed)

    # build dataset
    data_loader_train = build_dataloader(args, partition='train')
    data_loader_val   = build_dataloader(args, partition='test')

    # training svm
    import time
    st = time.time()

    # build train inputs
    input_set = []
    for batch_idx, (samples, targets) in enumerate(data_loader_train):
        if len(input_set) > 1000: break
        
        for sample, target in zip(samples, targets):
            if sample['events'] is None: continue

            bboxes = detect(sample, target)
            if len(bboxes) == 0: continue

            input_set.append({
                'feats': extract(sample, target, bboxes),
                'label': (target['labels'][0] == 0).long().item()
            })

    print(time.time() - st)

    # training a svm
    st = time.time()
    X_train = [v['feats'] for v in input_set]
    Y_train = [v['label'] for v in input_set]
    svm = SVC(C=1.0, kernel='linear', gamma='auto')
    svm.fit(X_train, Y_train)
    print(time.time() - st)
    
    # validation
    eval = Evaluator(aet_ids=data_loader_val.dataset.aet_ids, cat_ids={'fire':0})
    for batch_idx, (samples, targets) in enumerate(data_loader_val):
        outputs = []

        for sample, target in zip(samples, targets):
            if sample['events'] is None: 
                outputs.append({
                    'bboxes':torch.tensor([[]]),
                    'labels':torch.tensor([]),
                    'scores':torch.tensor([])
                })
                continue

            bboxes = detect(sample, target)
            if len(bboxes) == 0:
                outputs.append({
                    'bboxes':torch.tensor([[]]),
                    'labels':torch.tensor([]),
                    'scores':torch.tensor([])
                })
                continue
            
            if (svm.predict([extract(sample, target, bboxes)])[0] == 0):
                outputs.append({
                    'bboxes':torch.tensor(bboxes),
                    'labels':torch.tensor([0.]),
                    'scores':torch.tensor([1.])
                })
            else:
                outputs.append({
                    'bboxes':torch.tensor([[]]),
                    'labels':torch.tensor([]),
                    'scores':torch.tensor([])
                })

        eval.update(outputs, targets)

    stats = eval.summarize()
    print('\n'.join(f'{info}: {value:.3f}' for info, value in stats))
