import numpy as np

import torch
import torch.nn as nn
from torchvision.ops import box_iou

from typing import Dict, Iterable
from collections import defaultdict


class Evaluator(object):
    def __init__(self, 
                 aet_ids: Dict = {'test00': 0, 'test01': 1, 'test02': 2}, 
                 cat_ids: Dict = {'fire': 0, 'book': 1, 'dog': 2, 'cat': 3, 'fly': 4, 'brush': 5}) -> None:
        # parameters
        self.aet_ids = aet_ids
        self.cat_ids = cat_ids
        self.areas = {
            'all':    [0 ** 2, 1e5 ** 2],
            'small':  [0 ** 2, 16 ** 2],
            'medium': [16 ** 2, 32 ** 2],
            'large':  [32 ** 2, 1e5 ** 2]
        }
        self.max_det = 100
        self.iou_thr = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.rec_thr = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)

        # statistic members
        self.stats = None
        self.eval = None
        self.ious = dict()
        self._gts = defaultdict(list)
        self._dts = defaultdict(list)

    @torch.no_grad()
    def update(self, outputs, targets):
        # convert to pycoco type
        for output, target in zip(outputs, targets):
            # mapping to image id
            image_id = self.aet_ids[target['file']]
            
            # obtain basic info
            width, height = target['resolution']

            # convert target
            for tgt_cls, tgt_box in zip(target['labels'], target['bboxes']):
                # parse elements
                category_id = tgt_cls.item()
                xtl, ytl, w, h = tgt_box.tolist()
                
                # emplace back
                self._gts[image_id, category_id].append({
                    'image_id': image_id,
                    'category_id': category_id,
                    'bbox': [xtl, ytl, xtl + w, ytl + h],
                    'area': (w * width) * (h * height)
                }) 

            # convert output
            for out_log, out_box in zip(output['logits'], output['bboxes']):
                # parse elements
                category_id = out_log.max(-1)[1].item()
                xtl, ytl, w, h = out_box.tolist()
                score = out_log.max(-1)[0].item()

                # emplace back
                self._dts[image_id, category_id].append({
                    'image_id': image_id,
                    'category_id': category_id,
                    'bbox': [xtl, ytl, xtl + w, ytl + h],
                    'area': (w * width) * (h * height),
                    'score': score
                })

    def evaluate(self):
        # calculate iou
        self.ious = {
            (aet_id, cat_id): self._compute_iou(aet_id, cat_id) \
                for aet_id in self.aet_ids.values()
                for cat_id in self.cat_ids.values()
        }

        # evaluate
        self.stats = np.asarray([
            self._evaluate_every(aet_id, cat_id, area) \
                for cat_id in self.cat_ids.values()
                for area   in self.areas.values()
                for aet_id in self.aet_ids.values()
        ]).reshape(len(self.cat_ids), len(self.areas), len(self.aet_ids))

    def _compute_iou(self, aet_id, cat_id):
        gt = self._gts[aet_id, cat_id]
        dt = self._dts[aet_id, cat_id]
        
        if len(gt) == 0 or len(dt) == 0:
            return []

        return box_iou(torch.tensor([d['bbox'] for d in dt]),
                       torch.tensor([g['bbox'] for g in gt])).numpy()
    
    def _evaluate_every(self, aet_id, cat_id, area):
        gt = self._gts[aet_id, cat_id]
        dt = self._dts[aet_id, cat_id]

        if len(gt) == 0 and len(gt) == 0:
            return None
        
        # filter out of range
        for g in gt:
            if (g['area'] < area[0] or g['area'] > area[1]):
                g['ignore'] = 1
            else:
                g['ignore'] = 0

        # sort ignore last
        gt_ind = np.argsort([g['ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gt_ind]

        # sort highest score first
        dt_ind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dt_ind[0:self.max_det]]

        # load computed ious
        if len(self.ious[aet_id, cat_id]) > 0:
            ious = self.ious[aet_id, cat_id][:, gt_ind]
        else:
            ious = self.ious[aet_id, cat_id]
        
        # initialize
        T = len(self.iou_thr)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T, G))
        dtm  = np.zeros((T, D))
        gtIg = np.array([g['ignore'] for g in gt])
        dtIg = np.zeros((T, D))

        if not len(ious) == 0:
            for t_ind, t in enumerate(self.iou_thr):
                for dt_ind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m   = -1
                    for gt_ind, g in enumerate(gt):
                        # if this gt already matched, continue
                        if gtm[t_ind, gt_ind] > 0: continue
                        
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gt_ind] == 1: break

                        # continue to next gt unless better match made
                        if ious[dt_ind, gt_ind] < iou: continue
                        
                        # if match successful and best so far, store appropriately
                        iou = ious[dt_ind, gt_ind]
                        m = gt_ind
                    
                    # if match made store id of match for both dt and gt
                    if m == -1: continue
                    dtIg[t_ind, dt_ind] = gtIg[m]
                    dtm[t_ind, dt_ind]  = 1 # need to test
                    gtm[t_ind, m]       = 1 # need to test

        # set unmatched detections outside of area range to ignore
        a = np.array([d['area'] < area[0] or d['area'] > area[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))

        # store results for given image and category
        return {
                'image_id':     aet_id,
                'category_id':  cat_id,
                'aRng':         area,
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def accumulate(self):
        assert self.stats is not None, "Please run evaluate() first."

        # initialize
        T           = len(self.iou_thr)
        R           = len(self.rec_thr)
        K           = len(self.cat_ids)
        A           = len(self.areas)
        precision   = -np.ones((T, R, K, A)) # -1 for the precision of absent categories
        recall      = -np.ones((T, K, A))
        scores      = -np.ones((T, R, K, A))

        for cat_ind in range(K):
            for area_ind in range(A):
                # filter none elements
                E = [s for s in self.stats[cat_ind, area_ind, :] if s is not None]
                if len(E) == 0: continue

                # obtain sorted scores
                dtScores = np.concatenate([s['dtScores'][0:self.max_det] for s in E])
                inds = np.argsort(-dtScores, kind='mergesort')
                dtScores = dtScores[inds]

                dtm  = np.concatenate([e['dtMatches'][:, 0:self.max_det] for e in E], axis=1)[:,inds]
                dtIg = np.concatenate([e['dtIgnore'][:, 0:self.max_det]  for e in E], axis=1)[:,inds]
                gtIg = np.concatenate([e['gtIgnore'] for e in E])
                npig = np.count_nonzero(gtIg == 0)
                if npig == 0:
                    continue
                tps = np.logical_and(               dtm,  np.logical_not(dtIg))
                fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    nd = len(tp)
                    rc = tp / npig
                    pr = tp / (fp + tp + np.spacing(1))
                    q  = np.zeros((R,))
                    ss = np.zeros((R,))

                    if nd:
                        recall[t, cat_ind, area_ind] = rc[-1]
                    else:
                        recall[t, cat_ind, area_ind] = 0

                    # numpy is slow without cython optimization for accessing elements
                    # use python array gets significant speed improvement
                    pr = pr.tolist(); q = q.tolist()

                    for i in range(nd-1, 0, -1):
                        if pr[i] > pr[i-1]:
                            pr[i-1] = pr[i]

                    inds = np.searchsorted(rc, self.rec_thr, side='left')
                    try:
                        for ri, pi in enumerate(inds):
                            q[ri] = pr[pi]
                            ss[ri] = dtScores[pi]
                    except:
                        pass

                    precision[t, :, cat_ind, area_ind] = np.array(q)
                    scores[t, :, cat_ind, area_ind] = np.array(ss)

        self.eval = {
            'counts': [T, R, K, A],
            'precision': precision,
            'recall': recall,
            'scores': scores,
            'params': None
        }

    def summarize(self):
        self._summarize(1)
        self._summarize(1, iouThr=.5)
        self._summarize(1, iouThr=.75)
        self._summarize(1, areaRng='small')
        self._summarize(1, areaRng='medium')
        self._summarize(1, areaRng='large')
        self._summarize(0, areaRng='small')
        self._summarize(0, areaRng='medium')
        self._summarize(0, areaRng='large')

    def _summarize(self, ap=1, iouThr=None, areaRng='all'):
        iStr = ' {:<18} {} @ [ IoU = {:<9} | area = {:>6s} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap==1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(self.iou_thr[0], self.iou_thr[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        area_ind = [i for i, area in enumerate(self.areas.keys()) if area == areaRng]
        if ap == 1:
            # dimension of precision: [T x R x K x A]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == self.iou_thr)[0]
                s = s[t]
            s = s[..., area_ind]
        else:
            # dimension of recall: [T x K x A]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == self.iou_thr)[0]
                s = s[t]
            s = s[..., area_ind]
        if len(s[s > -1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, mean_s))
        return mean_s
        

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


if __name__ == '__main__':
    torch.manual_seed(114)

    num_query = 10
    num_class = 6

    # targets = tuple([
    #     {
    #         'file': 'test01',
    #         'labels': torch.randint(low=0, high=num_class, size=(28,)).cuda(),
    #         'bboxes': torch.rand([28, 4]).cuda(),
    #         'resolution': (346, 260)
    #     },
    #     {
    #         'file': 'test02',
    #         'labels': torch.randint(low=0, high=num_class, size=(13,)).cuda(),
    #         'bboxes': torch.rand([13, 4]).cuda(),
    #         'resolution': (346, 260)
    #     },
    # ])

    # outputs = tuple([
    #     {
    #         'logits': torch.rand([num_query, num_class + 1]).softmax(-1).cuda(),
    #         'bboxes': torch.rand([num_query, 4]).cuda(),
    #     },
    #     {
    #         'logits': torch.rand([num_query, num_class + 1]).softmax(-1).cuda(),
    #         'bboxes': torch.rand([num_query, 4]).cuda(),
    #     },
    # ])


    targets = tuple([
        {
            'file': 'test01',
            'labels': torch.tensor([0, 0]).cuda(),
            'bboxes': torch.tensor([[0.1, 0.2, 0.2, 0.1],
                                    [0.0, 0.0, 0.3, 0.3]]).cuda(),
            'resolution': (346, 260)
        }
    ])

    outputs = tuple([
        {
            'logits': torch.tensor([[0.3, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1]]).cuda(),
            'bboxes': torch.tensor([[0.0, 0.0, 0.4, 0.4],
                                    [0.1, 0.2, 0.1, 0.1],
                                    [0.2, 0.3, 0.1, 0.2],
                                    [0.1, 0.1, 0.8, 0.8],
                                    [0.5, 0.6, 0.3, 0.2],
                                    [0.7, 0.6, 0.1, 0.1]]).cuda(),
        }
    ])

    eval = Evaluator()
    eval.update(outputs, targets)
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    