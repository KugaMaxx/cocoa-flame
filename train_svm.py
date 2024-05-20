import cv2
import math
import torch
import argparse
import numpy as np
from sklearn.svm import SVC

from utils.misc import set_seed
from utils.eval import Evaluator
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
    # rescale
    box_x = bboxes[0][0] * target['resolution'][0]
    box_y = bboxes[0][1] * target['resolution'][1]
    box_w = bboxes[0][2] * target['resolution'][0]
    box_h = bboxes[0][3] * target['resolution'][1]
    
    # filter
    idn = np.logical_and(
        np.logical_and(sample['events'][:, 1] >= box_x, sample['events'][:, 1] <= (box_x + box_w)),
        np.logical_and(sample['events'][:, 2] >= box_y, sample['events'][:, 2] <= (box_y + box_h))
    ).bool()
    sample['events'] = sample['events'][idn]

    # predefine functions
    def project(events, size):
        image = np.zeros(size)
        image[events[:, 1], events[:, 2]] = 255
        return image.astype(np.uint8)

    def mcontour(count):
        contours = cv2.findContours(count, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        
        return contours[0]

    def moment(contour):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX = 0
            cY = 0

        return cX, cY

    # project
    count = project(sample['events'], target['resolution'])

    # contours
    contour = mcontour(count)
    area = cv2.contourArea(contour)
    arc_length = cv2.arcLength(contour, True) + 1

    # corners
    harris = cv2.cornerHarris(count, blockSize=2, ksize=3, k=0.04)

    # wrap events into buffers
    buffer = np.split(
        sample['events'],
        np.searchsorted(sample['events'][:, 0] - sample['events'][0, 0], [11000, 22000])
    )
    buffer = [buf for buf in buffer if len(buf) != 0]
    buf_contours = [
        mcontour(project(buffer, target['resolution'])) for buffer in buffer
    ]

    # buffer areas
    buf_areas = [
        cv2.contourArea(buf_contour) + 1 for buf_contour in buf_contours 
    ]

    # buffer movements
    buf_moments = np.array([
        moment(buf_countour) for buf_countour in buf_contours
    ])
    buf_movements = np.gradient(buf_moments, axis=0) if len(buf_moments) > 1 else np.array([[0, 0]])

    feat = [
        len(sample['events']),  # 事件输出率
        box_w / box_h,          # 长宽比
        area / (box_w * box_h), # 矩形度
        4 * math.pi * area / arc_length ** 2, # 圆形度
        (harris != 0).sum(), # 角点
        abs(buf_areas[-1] - buf_areas[0]) * len(buf_areas) / sum(buf_areas), # 面积变化率
        (buf_movements ** 2).sum() # 质心移动
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
        if len(input_set) > 600: break
        
        for sample, target in zip(samples, targets):
            if sample['events'] is None: continue

            bboxes = detect(sample, target)
            if len(bboxes) == 0: continue

            input_set.append({
                'feats': extract(sample, target, bboxes),
                'label': target['labels'][0].item()
            })

    print(time.time() - st)

    # training a svm
    st = time.time()
    X_train = [v['feats'] for v in input_set]
    Y_train = [v['label'] for v in input_set]
    svm = SVC(C=1.0, kernel='rbf', gamma='auto')
    svm.fit(X_train, Y_train)
    print(time.time() - st)
    
    # validation
    fp = 0
    eval = Evaluator(aet_ids=data_loader_val.dataset.aet_ids, cat_ids={'Flame': 0})
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
            
            feats = [extract(sample, target, bboxes)]
            pred_cls = svm.predict(feats)[0]
            outputs.append({
                'bboxes':torch.tensor([box.tolist() for box in bboxes]),
                'labels':torch.tensor([pred_cls]),
                'scores':torch.tensor([1.])
            })

            if (target['labels'][0] == 0 and pred_cls != 0): fp = fp + 1

            # print(feats, pred_cls, target['file'])
            # if (target['labels'][0] == 0 and pred_cls != 0):
            #     fp = fp + 1
            #     from utils.plot import plot_detection_result, plot_rescaled_image, plot_projected_events
            #     image  = np.zeros((260, 346)) if sample['frames'] is None else sample['frames'].numpy()
            #     events = np.zeros((1, 4))     if sample['events'] is None else sample['events'].numpy()
            #     image = plot_projected_events(image, events)
            #     image = plot_rescaled_image(image)
            #     image = plot_detection_result(image, 
            #                                 bboxes=(target['bboxes']).tolist(),
            #                                 labels=(target['labels']).tolist(),
            #                                 categories=data_loader_val.dataset.cat_ids,
            #                                 colors=[(0, 0, 255)])
                
            #     image = plot_detection_result(image, 
            #                                 bboxes=outputs[-1]['bboxes'],
            #                                 labels=outputs[-1]['labels'],
            #                                 scores=outputs[-1]['scores'],
            #                                 categories=data_loader_val.dataset.cat_ids,
            #                                 colors=[(255, 0, 0)])
            #     cv2.imwrite(f'./count_{fp:05d}.png', image)

        eval.update(outputs, targets)

    stats = eval.summarize()
    print('\n'.join(f'{info}: {value:.3f}' for info, value in stats))
    print(fp)
