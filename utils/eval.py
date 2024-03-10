import numpy as np

from typing import Dict, Iterable
from collections import defaultdict

import torch
from torchvision.ops import box_iou

from pycocotools.cocoeval import Params
import pycocotools.mask as maskUtils


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

        self.params = Params("bbox")

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
                    'bbox': [xtl, ytl, w, h],
                    'area': (w * width) * (h * height),
                    'iscrowd': 0,
                    'ignore': 0
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
                    'bbox': [xtl, ytl, w, h],
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
        maxDet = self.params.maxDets[-1]
        self.stats = np.asarray([
            self.evaluateImg(aet_id, cat_id, area, maxDet) \
                for cat_id in self.cat_ids.values()
                for area   in self.areas.values()
                for aet_id in self.aet_ids.values()
        ])
        # .reshape(len(self.cat_ids), len(self.areas), len(self.aet_ids))

    def _compute_iou(self, aet_id, cat_id):
        gt = self._gts[aet_id, cat_id]
        dt = self._dts[aet_id, cat_id]
        
        if len(gt) == 0 or len(dt) == 0:
            return []

        g = [g['bbox'] for g in gt]
        d = [d['bbox'] for d in dt]
        iscrowd = [int(o['iscrowd']) for o in gt]
        
        return maskUtils.iou(d, g, iscrowd)

        # return box_iou(torch.tensor([d['bbox'] for d in dt]),
        #                torch.tensor([g['bbox'] for g in gt])).numpy()

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        计算本张图片，特定类别，特定面积阈值，特定最大检测结果下的result。
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            # 本张图片特定类别的所有检测结果与GT
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None
 
        for g in gt:
            #如果不符合特定面积的阈值，就忽略
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0
 
        # sort dt highest score first, sort gt ignore last
        # gtind 前面都是 ignore为0 的gt 后面都是 ignore为1的gt
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        #挑出满足我们这个特定area阈值下的所有gt
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        #按照置信度大小挑出满足这个最大检测个数下的所有dt
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
 
        #得到满足area阈值的gt与所有dt的iou结果 （M * n（gtind））
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]
        #得到我们需要设置的IoU阈值，超过定义为正样本，不符合则为负样本
        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        #在每个阈值下的Gt是否得到匹配
        gtm  = np.zeros((T,G))
        #在每个阈值下的Dt是否得到匹配
        dtm  = np.zeros((T,D))
        #所有忽略的gt
        gtIg = np.array([g['_ignore'] for g in gt])
        #所有忽略的dt
        dtIg = np.zeros((T,D))
 
        #如果这张图片存在这个类别的gt与dt
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs): #IoU index， IoU阈值
                #按照置信度大小排序好的前 max_Det个dt
                for dind, d in enumerate(dt):
                    # 如果m= -1 代表这个dt没有得到匹配 m代表dt匹配的最好的gt的下标
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # 如果这个gt已经被其他置信度更好的dt匹配到了，本轮的dt就不能匹配这个gt了。
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # 因为gt已经按照ignore排好序了，前面的为0，于是当我们碰到第一个gt的ignore为1时，判断这个dt是否已经匹配到了
                        #其他的gt，如果m>-1证明并且m对应的gt没有被ignore，就直接结束即可，对应的就是这个dt最好的gt。
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # 如果计算dt与gt的iou小于目前最佳的IoU，忽略这个gt
                        if ious[dind,gind] < iou:
                            continue
                        # 超过当前最佳的IoU，更新IoU与m的值
                        iou=ious[dind,gind]
                        m=gind
                    # 如果这个dt没有对应的gt与其匹配，继续dt的下一个循环
                    if m ==-1:
                        continue
                    # 把当前dt与第m个gt进行匹配，修改dtm与gtm的值，分别一一对应
                    dtIg[tind,dind] = gtIg[m] # 如果这个dt对应的最佳gt本身就是被ignore的，就把这个dt也设置为ignore。
                    dtm[tind,dind]  = 1
                    gtm[tind,m]     = 1
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }
 
    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]

        T           = len(self.iou_thr)
        R           = len(self.rec_thr)
        K           = len(self.cat_ids)
        A           = len(self.areas)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))
 
        # get inds to evaluate
        k_list = [n for n, k in enumerate(self.cat_ids.values())] #对应不重复的K的id list 后续同此
        m_list = [m for n, m in enumerate(p.maxDets)]
        a_list = [n for n, a in enumerate(self.areas.values())]
        i_list = [n for n, i in enumerate(self.aet_ids.values())]
        I0 = len(self.aet_ids) #多少个图片
        A0 = len(self.areas) #多少个面积阈值
        # retrieve E at each category, area range, and max number of detections
        # self.stats 索引顺序是 K,A,M,I 所以找到在特定K，A，M下的所有图片，需要按照如下的三维索引
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0  # 当前K0前面过了多少图片与面积阈值
            for a, a0 in enumerate(a_list):
                Na = a0*I0 #在当前K0前面过了多少阈值
                for m, maxDet in enumerate(m_list):
                    #k0，a0下的所有Images
                    E = [self.stats[Nk + Na + i] for i in i_list]
                    E = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue
                    #k0，a0，maxdet下的所有Images的得分
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])
 
                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    # k0，a0，maxdet下所有Images得分从高到底的索引 inds
                    inds = np.argsort(-dtScores, kind='mergesort')
                    #按照得分从高到低排序
                    dtScoresSorted = dtScores[inds]
                    # 在当前k0,a0下，每张图片不超过MaxDet的所有det按照ind排序。 dtm[T,sum(Det) in every imges]
                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    #有多少个正样本
                    npig = np.count_nonzero(gtIg==0 )
                    if npig == 0:
                        continue
                    # 如果dtm对应的匹配gt不为0，且对应的gt没有被忽略，这个dt就是TP tips:[1,0,1,0,1,0]
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    #dtm对应的gt为0， 并且这个dt也没有被忽略，这个dt就是FP  tips:[0,1,0,1,0,1]
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )
 
                    # 按照行的方式（每个Iou阈值下）进行匹配到的累加 每个index也就是到这个置信度的时候有多少个tp，有多少个fp
                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float64)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float64)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp) #得到这个Iou下对应的tp tips:[1,0,2,0,3,0]
                        fp = np.array(fp) #得到这个IoU下对应的fp tips:[0,1,0,2,0,3]
                        nd = len(tp) #有多少个tp
                        rc = tp / npig #每个置信度分数下对应的recall 如上述例子 若有3个正样本 则rc=[1/3,1/3,2/3,2/3,1,1]
                        pr = tp / (fp+tp+np.spacing(1)) #每个阶段对应的精度
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))
 
                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0
 
                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()
 
                        #当前i下的最大精度
                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]
 
                        #找到每个recall发生变化的时候的index，与p.recThrs一一对应，最接近其的值的index
                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                #得到每个recall阈值对应的最大精度，存入q中
                                q[ri] = pr[pi]
                                #得到这个recall值下的得分
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q) # 按照recall的大小存入对应的精度
                        scores[t,:,k,a,m] = np.array(ss) #存入对应的分数
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
 
    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)
            # 如果是'all' 就是所有尺度， 如果不是就是特定的尺度
            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            # 如果是ap，就从precision中得到对应面积阈值、最大检测数下的精度
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # 得到特定IoU下的所有pr
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
 
            # 如果是recall，就取出recall的值
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            #除去-1 其他的计算平均精度
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1) # all iouThr， 所有recall下，所有面积下， 所有类别，在最大检测数100下的的平均AP
            # [1]:IoU阈值为0.5 所有recall下，所有面积下， 所有类别，在最大检测数100下的的平均AP
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            # [2]:IoU阈值为0.75 所有recall下，所有面积下， 所有类别，在最大检测数100下的的平均AP
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            #[3]: all iouThr， 所有recall下，small面积下， 所有类别，在最大检测数100下的的平均AP
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            #[4]: all iouThr， 所有recall下，medium面积下， 所有类别，在最大检测数100下的的平均AP
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            #[5]: all iouThr， 所有recall下，large面积下， 所有类别，在最大检测数100下的的平均AP
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            #[6]: all iouThr，所有面积下， 所有类别，在最大检测数1下的的平均recall
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            #[7]: all iouThr，所有面积下， 所有类别，在最大检测数10下的的平均recall
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            # [8]: all iouThr，所有面积下， 所有类别，在最大检测数100下的的平均recall
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            #[9]: all iouThr，small面积下， 所有类别，在最大检测数100下的的平均recall
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            # [10]: all iouThr，medium面积下， 所有类别，在最大检测数100下的的平均recall
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            # [11]: all iouThr，large面积下， 所有类别，在最大检测数100下的的平均recall
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    # def _evaluate_every(self, aet_id, cat_id, area):
    #     gt = self._gts[aet_id, cat_id]
    #     dt = self._dts[aet_id, cat_id]

    #     if len(gt) == 0 and len(gt) == 0:
    #         return None
        
    #     # filter out of range
    #     for g in gt:
    #         if (g['area'] < area[0] or g['area'] > area[1]):
    #             g['ignore'] = 1
    #         else:
    #             g['ignore'] = 0

    #     # sort ignore last
    #     gt_ind = np.argsort([g['ignore'] for g in gt], kind='mergesort')
    #     gt = [gt[i] for i in gt_ind]

    #     # sort highest score first
    #     dt_ind = np.argsort([-d['score'] for d in dt], kind='mergesort')
    #     dt = [dt[i] for i in dt_ind[0:self.max_det]]

    #     # load computed ious
    #     if len(self.ious[aet_id, cat_id]) > 0:
    #         ious = self.ious[aet_id, cat_id][:, gt_ind]
    #     else:
    #         ious = self.ious[aet_id, cat_id]
        
    #     # initialize
    #     T = len(self.iou_thr)
    #     G = len(gt)
    #     D = len(dt)
    #     gtm  = np.zeros((T, G))
    #     dtm  = np.zeros((T, D))
    #     gtIg = np.array([g['ignore'] for g in gt])
    #     dtIg = np.zeros((T, D))

    #     if not len(ious) == 0:
    #         for t_ind, t in enumerate(self.iou_thr):
    #             for dt_ind, d in enumerate(dt):
    #                 # information about best match so far (m=-1 -> unmatched)
    #                 iou = min([t, 1 - 1e-10])
    #                 m   = -1
    #                 for gt_ind, g in enumerate(gt):
    #                     # if this gt already matched, continue
    #                     if gtm[t_ind, gt_ind] > 0: continue
                        
    #                     # if dt matched to reg gt, and on ignore gt, stop
    #                     if m > -1 and gtIg[m] == 0 and gtIg[gt_ind] == 1: break

    #                     # continue to next gt unless better match made
    #                     if ious[dt_ind, gt_ind] < iou: continue
                        
    #                     # if match successful and best so far, store appropriately
    #                     iou = ious[dt_ind, gt_ind]
    #                     m = gt_ind
                    
    #                 # if match made store id of match for both dt and gt
    #                 if m == -1: continue
    #                 dtIg[t_ind, dt_ind] = gtIg[m]
    #                 dtm[t_ind, dt_ind]  = 1 # need to test
    #                 gtm[t_ind, m]       = 1 # need to test

    #     # set unmatched detections outside of area range to ignore
    #     a = np.array([d['area'] < area[0] or d['area'] > area[1] for d in dt]).reshape((1, len(dt)))
    #     dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))

    #     # store results for given image and category
    #     return {
    #             'image_id':     aet_id,
    #             'category_id':  cat_id,
    #             'aRng':         area,
    #             'dtMatches':    dtm,
    #             'gtMatches':    gtm,
    #             'dtScores':     [d['score'] for d in dt],
    #             'gtIgnore':     gtIg,
    #             'dtIgnore':     dtIg,
    #         }

    # def accumulate(self):
    #     assert self.stats is not None, "Please run evaluate() first."

    #     # initialize
    #     T           = len(self.iou_thr)
    #     R           = len(self.rec_thr)
    #     K           = len(self.cat_ids)
    #     A           = len(self.areas)
    #     precision   = -np.ones((T, R, K, A)) # -1 for the precision of absent categories
    #     recall      = -np.ones((T, K, A))
    #     scores      = -np.ones((T, R, K, A))

    #     for cat_ind in range(K):
    #         for area_ind in range(A):
    #             # filter none elements
    #             E = [s for s in self.stats[cat_ind, area_ind, :] if s is not None]
    #             if len(E) == 0: continue

    #             # obtain sorted scores
    #             dtScores = np.concatenate([s['dtScores'][0:self.max_det] for s in E])
    #             inds = np.argsort(-dtScores, kind='mergesort')
    #             dtScores = dtScores[inds]

    #             dtm  = np.concatenate([e['dtMatches'][:, 0:self.max_det] for e in E], axis=1)[:,inds]
    #             dtIg = np.concatenate([e['dtIgnore'][:, 0:self.max_det]  for e in E], axis=1)[:,inds]
    #             gtIg = np.concatenate([e['gtIgnore'] for e in E])
    #             npig = np.count_nonzero(gtIg == 0)
    #             if npig == 0:
    #                 continue
    #             tps = np.logical_and(               dtm,  np.logical_not(dtIg))
    #             fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

    #             tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
    #             fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
    #             for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
    #                 tp = np.array(tp)
    #                 fp = np.array(fp)
    #                 nd = len(tp)
    #                 rc = tp / npig
    #                 pr = tp / (fp + tp + np.spacing(1))
    #                 q  = np.zeros((R,))
    #                 ss = np.zeros((R,))

    #                 if nd:
    #                     recall[t, cat_ind, area_ind] = rc[-1]
    #                 else:
    #                     recall[t, cat_ind, area_ind] = 0

    #                 # numpy is slow without cython optimization for accessing elements
    #                 # use python array gets significant speed improvement
    #                 pr = pr.tolist(); q = q.tolist()

    #                 for i in range(nd-1, 0, -1):
    #                     if pr[i] > pr[i-1]:
    #                         pr[i-1] = pr[i]

    #                 inds = np.searchsorted(rc, self.rec_thr, side='left')
    #                 try:
    #                     for ri, pi in enumerate(inds):
    #                         q[ri] = pr[pi]
    #                         ss[ri] = dtScores[pi]
    #                 except:
    #                     pass

    #                 precision[t, :, cat_ind, area_ind] = np.array(q)
    #                 scores[t, :, cat_ind, area_ind] = np.array(ss)

    #     self.eval = {
    #         'counts': [T, R, K, A],
    #         'precision': precision,
    #         'recall': recall,
    #         'scores': scores,
    #         'params': None
    #     }

    # def summarize(self):
    #     self._summarize(1)
    #     self._summarize(1, iouThr=.5)
    #     self._summarize(1, iouThr=.75)
    #     self._summarize(1, areaRng='small')
    #     self._summarize(1, areaRng='medium')
    #     self._summarize(1, areaRng='large')
    #     self._summarize(0, areaRng='small')
    #     self._summarize(0, areaRng='medium')
    #     self._summarize(0, areaRng='large')

    # def _summarize(self, ap=1, iouThr=None, areaRng='all'):
    #     iStr = ' {:<18} {} @ [ IoU = {:<9} | area = {:>6s} ] = {:0.3f}'
    #     titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
    #     typeStr = '(AP)' if ap==1 else '(AR)'
    #     iouStr = '{:0.2f}:{:0.2f}'.format(self.iou_thr[0], self.iou_thr[-1]) \
    #         if iouThr is None else '{:0.2f}'.format(iouThr)

    #     area_ind = [i for i, area in enumerate(self.areas.keys()) if area == areaRng]
    #     if ap == 1:
    #         # dimension of precision: [T x R x K x A]
    #         s = self.eval['precision']
    #         # IoU
    #         if iouThr is not None:
    #             t = np.where(iouThr == self.iou_thr)[0]
    #             s = s[t]
    #         s = s[..., area_ind]
    #     else:
    #         # dimension of recall: [T x K x A]
    #         s = self.eval['recall']
    #         if iouThr is not None:
    #             t = np.where(iouThr == self.iou_thr)[0]
    #             s = s[t]
    #         s = s[..., area_ind]
    #     if len(s[s > -1])==0:
    #         mean_s = -1
    #     else:
    #         mean_s = np.mean(s[s > -1])
    #     print(iStr.format(titleStr, typeStr, iouStr, areaRng, mean_s))
    #     return mean_s


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
