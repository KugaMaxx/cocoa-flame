import time
import argparse
import dv_processing as dv
import dv_toolkit as kit
import numpy as np
import math
import torch
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from utils.eval import Evaluator
from datasets import build_dataloader
from utils.misc import set_seed
from models.scout import flame_scout
from utils.plot import plot_detection_result, plot_projected_events, plot_rescaled_image



def parse_args():
    parser = argparse.ArgumentParser()

    # common
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--output_dir', default='./checkpoint', type=str)
    
    # training strategy
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--learning_rate', default=1e-5, type=float)

    # dataset
    parser.add_argument('--dataset_file', default='dv_fire')
    parser.add_argument('--dataset_path', default='/home/dszh/Workspace/cocoa-flame/datasets/dv_fire/aedat_to_data/', type=str)
    parser.add_argument('--num_workers', default=1, type=int)

    # model
    parser.add_argument('--model_name', default='point_mlp', type=str)

    ## point_mlp

    ## local grouper

    ## criterion

    # checkpoint
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint_dir', default='./checkpoint', type=str)
    # parser.add_argument('--cpkt_name', default='last_checkpoint', type=str)
    parser.add_argument('--checkpoint_epoch', default=50, type=int)

    return parser.parse_args()


def to_do_train(samples, targets):
    results=[]
    for sample, target in zip(samples, targets):
        if sample["events"] is None: 
            results.append([])

        # construct events
        else:
            events = kit.EventStorage()
            for timestamp, x, y, polarity in sample['events']:
            
               events.emplace_back(timestamp, x, y, polarity)
            # run model
            model = flame_scout.init(target['resolution'])
            model.accept(events)
            results.append(model.detect())
    return results

# opencv 可视化
def fill(events):
    matrix2d = np.zeros((260, 346))
    if events is None:
        return matrix2d
    matrix2d[events[:,2],events[:,1]]=255
    return matrix2d

# 主体边界
def boundary(border):
    area0 = cv2.contourArea(border[0])
    border0 = border[0]
    for j, i in enumerate(border):
        if cv2.contourArea(i)>area0:
            area0 = cv2.contourArea(i)
            border0 = i
    return area0, border0

# 膨胀腐蚀保留主体(暂时不需要)
def solve(matrix2d,events):
    imgErode = cv2.erode(matrix2d.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    imgDilate = cv2.dilate(imgErode, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    contours, hierarchy = cv2.findContours(imgDilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imgRGB_dilate = cv2.cvtColor(imgDilate, cv2.COLOR_GRAY2RGB)
    square_main, contours_main = boundary(contours)
    contours_main=(contours_main)
    matrix_main = np.zeros((260, 346))
    for j in events:
      
         idx=cv2.pointPolygonTest(contours_main, (int(j[1]), int(j[2])), False)
         if idx>0:
            matrix_main[j[2]][j[1]]=255
    imgRGB=cv2.cvtColor(matrix_main.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    imgRGB_main = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)    
    return imgRGB_main

# 非极大值抑制
def corner_nms(corner, kernal=5):
        out = corner.copy()
        row_s = int(kernal / 2)
        row_e = out.shape[0] - int(kernal / 2)
        col_s, col_e = int(kernal / 2), out.shape[1] - int(kernal / 2)
        for r in range(row_s, row_e):
            for c in range(col_s, col_e):
                if corner[r, c] == 0:  # 不是可能的角点
                    continue
                zone = corner[r - int(kernal / 2):r + int(kernal / 2) + 1, c - int(kernal / 2):c + int(kernal / 2) + 1]
                index = corner[r, c] < zone
                (x, y) = np.where(index == True)
                if len(x) > 0:  # 说明corner[r,c]不是最大，直接归零将其抑制
                    out[r, c] = 0
        return out

# 角点总数
def sum_corners(imgRGB_main):
    dst = cv2.cornerHarris(src=imgRGB_main, blockSize=2, ksize=3, k=0.04)
    dst1 = dst.copy()
    dst1[dst <= 0.01 * dst.max()] = 0
    score_nms = corner_nms(dst1)
    num_corners = np.sum(score_nms != 0)
    img=cv2.cvtColor(imgRGB_main, cv2.COLOR_GRAY2RGB)
    img[score_nms != 0]=(0,0,255)
    cv2.imwrite('./out.png',img)
    return num_corners

def expand(rect):
    minx=rect[0]*346
    width=rect[2]*346
    miny=rect[1]*260
    height=rect[3]*260
    return [minx,miny,width,height]

# 候选框事件输出率
def event_output(rect,events):
    if events is None:
            return 0
    output=0
    idx = np.logical_and(events[:, 1] >=rect[0], events[:, 1] <= (rect[0]+rect[2]))
    idy = np.logical_and(events[:,2]>=rect[1],events[:,2]<=(rect[1]+rect[3]))
    id = np.logical_and(idx,idy)
    output = id.sum() 
    return (output)

# 候选框事件长宽比
def length_width(rect,events):
    if events is None:
        return 0
    return (rect[2]/(rect[3]))

# 候选框内角点数
def corner_in_box(rect,img):
    dst = cv2.cornerHarris(src=img, blockSize=2, ksize=3, k=0.04)
    dst1 = dst.copy()
    dst1[dst <= 0.01 * dst.max()] = 0
    score_nms = corner_nms(dst1) 
    score_num=score_nms[int(rect[1]+1):int(rect[1]+rect[3]),int(rect[0]+1):int(rect[0]+rect[2])]
    num=np.sum(score_num!=0)
    return num

# 矩形度
def rectangularity(rect,events):
    area=square(rect,events)
    return area/(rect[2]*rect[3])

# 圆形度
def circle(rect,events):
    matrix2d=np.zeros((260,346))
    mask=matrix2d.copy()
    matrix2d[events[:,2],events[:,1]]=255
    mask[int(rect[1]+1):int(rect[1]+rect[3]),int(rect[0]+1):int(rect[0]+rect[2])]=1
    matrix2d=matrix2d*mask
    contours, hierarchy = cv2.findContours(matrix2d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    square_main, contours_main = boundary(contours)
    if square_main==0:
        return 0
    length = cv2.arcLength(contours_main, True)
    roundness=4*math.pi*square_main/(length*length)
    return roundness

# 计算框内面积
def square(rect,events):
    matrix2d=np.zeros((260,346))
    mask=matrix2d.copy()
    matrix2d[events[:,2],events[:,1]]=255
    mask[int(rect[1]+1):int(rect[1]+rect[3]),int(rect[0]+1):int(rect[0]+rect[2])]=1
    matrix2d=matrix2d*mask
    contours, hierarchy = cv2.findContours(matrix2d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area=np.sum(list(map(cv2.contourArea,contours)))
    return area

# 面积变化率
def areachange(rect,events):
    # 对齐时间
    timestamps = events[:, 0] - events[0, 0]
    ids = np.searchsorted(timestamps, [11000, 22000])
    # 按索引分割
    events1 = events[0:ids[0], :]
    events2 = events[ids[0]:ids[1], :]
    events3 = events[ids[1]:, :]
    area1=square(rect,events1)
    area2=square(rect,events2)
    area3=square(rect,events3)
    # 有改动的时候会出现这个圆
    return (area3-area1)*3/(area1+area2+area3)

def dict_list(contour):
    return list(cv2.moments(contour).values())
# 形心移动
def move(rect,events):
    timestamps = events[:, 0] - events[0, 0]
    ids = np.searchsorted(timestamps, [11000, 22000])
    events1 = events[0:ids[0], :]
    events3 = events[ids[1]:, :]
    matrix2d1=np.zeros((260,346))
    matrix2d3=matrix2d1.copy()
    mask=matrix2d1.copy()
    mask[int(rect[1]+1):int(rect[1]+rect[3]),int(rect[0]+1):int(rect[0]+rect[2])]=1
    matrix2d1[events1[:,2],events1[:,1]]=255
    matrix2d3[events3[:,2],events3[:,1]]=255
    matrix2d1=matrix2d1*mask
    matrix2d3=matrix2d3*mask
    num1=np.sum(matrix2d1)/255
    num3=np.sum(matrix2d3)/255
    if num1==0 or num3==0:
        return 0
    contours1, hierarchy1 = cv2.findContours(matrix2d1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours3, hierarchy = cv2.findContours(matrix2d3.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    moment1=list(map(dict_list,contours1))
    moment3=list(map(dict_list,contours3))
    moment1=torch.tensor(moment1)
    moment3=torch.tensor(moment3)
    sumx1=moment1[:,1].sum()/moment1[:,0].sum()
    sumx3=moment3[:,1].sum()/moment3[:,0].sum()
    sumy1=moment1[:,2].sum()/moment1[:,0].sum()
    sumy3=moment3[:,2].sum()/moment3[:,0].sum()
    return math.sqrt(math.pow((sumx3/num3-sumx1/num1),2)+math.pow((sumy3/num3-sumy1/num1),2))

# 傅里叶频闪
def fourier(rect,events):
    pass

# 训练集标签
def make_label(boxA,boxB,events):
    if events is None:
        return 0
    tl_x1, tl_y1, w1, h1 = boxA
    tl_x2, tl_y2, w2, h2 = boxB
   
    rb_x1, rb_y1 = tl_x1 + w1, tl_y1 + h1
    rb_x2, rb_y2 = tl_x2 + w2, tl_y2 + h2

    # 计算交集的坐标
    x_inter1 = max(tl_x1, tl_x2) #union的左上角x
    y_inter1 = max(tl_y1, tl_y2) #union的左上角y
    x_inter2 = min(rb_x1, rb_x2) #union的右下角x
    y_inter2 = min(rb_y1, rb_y2) #union的右下角y

    # 判断是否相交
    if x_inter2 < x_inter1 or y_inter2 < y_inter1:
        iou=0.0  # 框不相交，IOU为0
    else:
    # 计算交集部分面积
        interArea = max(0, x_inter2 - x_inter1) * max(0, y_inter2 - y_inter1)

    # 分别计算两个box的面积
        area_box1 = w1*h1
        area_box2 = w2*h2

    #计算IOU，交集比并集，并集面积=两个矩形框面积和-交集面积
        iou=interArea / (area_box1 + area_box2 - interArea)
    if iou>0.5:
       return 1
    else:
       return 0

if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # fix for reproducibility
    seed = set_seed(args.seed)

    # create logger
    logger = None

    # initialize
    stat = dict(
        epoch = 0, args = args,
        weight_decay  = args.weight_decay,
        learning_rate = args.learning_rate,
    )

    print("running")

    # build dataset
    data_loader_train = build_dataloader(args, partition='train') 
    data_loader_val   = build_dataloader(args, partition='test')

# new add ------------------------------------------------------------------
    import torch
    import dv_processing as dv
    import dv_toolkit as kit
    from datetime import timedelta
    from numpy.lib.recfunctions import structured_to_unstructured
    fns = []
    fns_events=[]
    def to_do_train_wo_label(data):
        if data['events'].isEmpty():
            return
        sample = {
                  "events": torch.from_numpy(structured_to_unstructured(data['events'].numpy())) \
                    if not data['events'].isEmpty()  else None,
                    "frames": torch.from_numpy(data['frames'].front().image) \
                    if not data['frames'].isEmpty()  else None,
                  }
        fns_events.append(sample['events'])
        model = flame_scout.init((346, 260))
        model.accept(data['events'])
        fns.append(model.detect())


    # load offline data
    reader = kit.io.MonoCameraReader(f"./tmp/UAV-ND00-2.aedat4")
    data, resolution = reader.loadData(), reader.getResolution("events")

   # do every 33ms (cannot modify!)
    slicer = kit.MonoCameraSlicer()
    slicer.doEveryTimeInterval("events", timedelta(milliseconds=33), to_do_train_wo_label)
    slicer.accept(data)

# new add --------------------------------------------------------------------------------------------
    twofire=[]
    twofire_events=[]
    def to_do_train_wo_label_twofire(data):
        if data['events'].isEmpty():
            return
        sample = {
                    "events": torch.from_numpy(structured_to_unstructured(data['events'].numpy())) \
                    if not data['events'].isEmpty()  else None,
                    "frames": torch.from_numpy(data['frames'].front().image) \
                    if not data['frames'].isEmpty()  else None,
                  }
        twofire_events.append(sample['events'])
        model = flame_scout.init((346, 260))
        model.accept(data['events'])
        twofire.append(model.detect())
    # load offline data
    reader = kit.io.MonoCameraReader(f"./tmp/S03_C05.aedat4")
    data, resolution = reader.loadData(), reader.getResolution("events")

   # do every 33ms (cannot modify!)
    slicer = kit.MonoCameraSlicer()
    slicer.doEveryTimeInterval("events", timedelta(milliseconds=33), to_do_train_wo_label_twofire)
    slicer.accept(data)
#  new add--------------------------------------------------------------------------------------------------
    # fire_people=[]
    # fire_people_events=[]
    # def to_do_train_wo_label_fire_people(data):
    #     if data['events'].isEmpty():
    #         return
    #     sample = {
    #               "events": torch.from_numpy(structured_to_unstructured(data['events'].numpy())) \
    #                 if not data['events'].isEmpty()  else None,
    #                 "frames": torch.from_numpy(data['frames'].front().image) \
    #                 if not data['frames'].isEmpty()  else None,
    #               }
    #     fire_people_events.append(sample['events'])
    #     model = flame_scout.init((346, 260))
    #     model.accept(data['events'])
    #     fire_people.append(model.detect())
    # # load offline data
    # reader = kit.io.MonoCameraReader(f"./tmp/Hybrid_02.aedat4")
    # data, resolution = reader.loadData(), reader.getResolution("events")

    # # do every 33ms (cannot modify!)
    # slicer = kit.MonoCameraSlicer()
    # slicer.doEveryTimeInterval("events", timedelta(milliseconds=33), to_do_train_wo_label_fire_people)
    # slicer.accept(data)
# new add----------------------------------------------------------------------------------------------------------

    # 单火焰
    X=[]
    Y=[]
    st = time.time()
    for batch_idx, (samples, targets) in enumerate(data_loader_train):
        if batch_idx>15:
            break
        result = to_do_train(samples, targets)
        for i,j in enumerate(result):
            matrix_init=fill(samples[i]['events'])
            tar=expand(targets[i]['bboxes'][0])
            X.append([event_output(tar,samples[i]['events']),length_width(tar,samples[i]['events']),rectangularity(tar,samples[i]['events']),circle(tar,samples[i]['events']),corner_in_box(tar,matrix_init.astype(np.uint8)),areachange(tar,samples[i]['events']),move(tar,samples[i]['events'])])
            Y.append(1)
    lenY=len(Y)
    print(f"单火焰推理时间：{time.time() - st}")
    
    # 双火焰
    for i,j in enumerate(twofire):
        if i>=lenY:
            break
        matrix_init=fill(twofire_events[i])
        tar=expand(j[0])
        X.append([event_output(tar,twofire_events[i]),length_width(tar,twofire_events[i]),rectangularity(tar,twofire_events[i]),circle(tar,twofire_events[i]),corner_in_box(tar,matrix_init.astype(np.uint8)),areachange(tar,twofire_events[i]),move(tar,twofire_events[i])])
        Y.append(1)
   
    # 运动物体
    for i,j in enumerate(fns):
        if Y.count(0)==Y.count(1):
            break
        if j==[]:
            X.append([0,0,0,0,0,0,0])
            Y.append(0)
            continue
        matrix_init=fill(fns_events[i])
        tar=expand(j[0])
        X.append([event_output(tar,fns_events[i]),length_width(tar,fns_events[i]),rectangularity(tar,fns_events[i]),circle(tar,fns_events[i]),corner_in_box(tar,matrix_init.astype(np.uint8)),areachange(tar,fns_events[i]),move(tar,fns_events[i])])
        Y.append(0)
        
    # train
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2 , random_state=0,shuffle=True)
    svm1=SVC(C=1.0,kernel='linear',degree=3,gamma='auto')
    svm2=SVC(C=1.0,kernel='rbf',degree=3,gamma='auto')
    svm1.fit(X_train, Y_train)
    svm2.fit(X_train, Y_train)
    Y_pred1=svm1.predict(X_test)
    Y_pred2=svm2.predict(X_test)
    print(metrics.accuracy_score(Y_test,Y_pred1))
    print(metrics.accuracy_score(Y_test,Y_pred2))
    
    # inspect
    # A=[]
    # C=[]
    # for i,j in enumerate(fire_people):
    #     if i>=100:
    #        break
    #     matrix_init=fill(fire_people_events[i])
    #     for n in range(len(j)):
    #         A.append([event_output(j[n],fire_people_events[i]),length_width(j[n],fire_people_events[i]),rectangularity(j[n],fire_people_events[i]),circle(j[n],fire_people_events[i]),corner_in_box(j[n],matrix_init.astype(np.uint8))])
    #         if n==0 or n==1:
    #             C.append(1)
    #         else:
    #             C.append(0)
    # B=svm1.predict(A)
    print("ending")    
    breakpoint()




    # Eval
    

    # total_loss = 0
    # X=[]
    # Y=[]
    # eval = Evaluator(aet_ids=data_loader_train.dataset.aet_ids, 
    #                  cat_ids=data_loader_train.dataset.cat_ids)
    # for batch_idx, (samples, targets) in enumerate(data_loader_train):
    #     outputs=[]
    #     if batch_idx>30:
    #         break
    #     result = to_do_train(samples, targets)
    #     for i in range(len(result)):
    #         outputs.append({'bboxes':torch.tensor([result[i][0]]),'labels':torch.tensor([0]),'scores':torch.tensor([1])})
    #     eval.update(outputs, targets)
    #     def plot(sample, target, output, i):
    #         import cv2
    #         import numpy as np
    #         image  = np.zeros((260, 346)) if sample['frames'] is None else sample['frames'].numpy()
    #         events = np.zeros((1, 4))     if sample['events'] is None else sample['events'].numpy()

    #         image = plot_projected_events(image, events)
    #         image = plot_rescaled_image(image)
    #         image = plot_detection_result(image, 
    #                                         bboxes=(target['bboxes']).tolist(),
    #                                         labels=(target['labels']).tolist(),
    #                                         colors=[(0, 0, 255)])
    #         idn = 6
    #         image = plot_detection_result(image, 
    #                                         bboxes=output['bboxes'][:idn],
    #                                         labels=output['labels'][:idn],
    #                                         scores=output['scores'][:idn],
    #                                         colors=[(255, 0, 0)])
    #         cv2.imwrite(f'./result_{i}.png', image)
    #         return True
        

    #     if batch_idx == 0:
    #         results = [plot(sample, target, output, i) \
    #                    for i, (sample, target, output) in enumerate(zip(samples, targets, outputs))]
    #         pass

    # eval.summarize()
    # breakpoint()

            # matrix_init=fill(samples[i]['events'])
            # if len(result[i])<2:
            #    random_rect=[np.random.randint(0, 1000)/1000,np.random.randint(0, 1000)/1000,targets[i]['bboxes'][0][2],targets[i]['bboxes'][0][3]]
            #    X.append([event_output(targets[i]['bboxes'][0],samples[i]['events']),length_width(targets[i]['bboxes'][0],samples[i]['events']),rectangularity(targets[i]['bboxes'][0],samples[i]['events']),corner_in_box(targets[i]['bboxes'][0],matrix_init.astype(np.uint8))])
            #    X.append([event_output(random_rect,samples[i]['events']),length_width(random_rect,samples[i]['events']),rectangularity(random_rect,samples[i]['events']),corner_in_box(random_rect,matrix_init.astype(np.uint8))])
            #    Y.append(make_label(targets[i]['bboxes'][0],targets[i]['bboxes'][0],samples[i]['events']))
            #    Y.append(make_label(random_rect,targets[i]['bboxes'][0],samples[i]['events']))
            # else:
            #    X.append([event_output(targets[i]['bboxes'][0],samples[i]['events']),length_width(targets[i]['bboxes'][0],samples[i]['events']),rectangularity(targets[i]['bboxes'][0],samples[i]['events']),corner_in_box(targets[i]['bboxes'][0],matrix_init.astype(np.uint8))])
            #    X.append([event_output(result[i][1],samples[i]['events']),length_width(result[i][1],samples[i]['events']),rectangularity(result[i][1],samples[i]['events']),corner_in_box(result[i][1],matrix_init.astype(np.uint8))])
            #    Y.append(make_label(targets[i]['bboxes'][0],targets[i]['bboxes'][0],samples[i]['events']))
            #    Y.append(make_label(result[i][1],targets[i]['bboxes'][0],samples[i]['events']))
           
