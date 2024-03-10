import argparse
from pathlib import Path
import dv_processing as dv
import dv_toolkit as kit
import numpy as np
import torch
import cv2
from torch.utils.tensorboard import SummaryWriter

from models import build_model
from datasets import build_dataloader
from utils.misc import set_seed, save_checkpoint, load_checkpoint
from engine import train, evaluate
from models.scout import flame_scout


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
            result.append([])

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
    else:
        for i in events:
           matrix2d[i[2]][i[1]]=255
        return matrix2d

# 主体边界
def boundary(border):
        area = [[0 for x in range(2)] for y in range(500)]
        j = 0
        for i in border:
            area[j][0] = cv2.contourArea(i)
            area[j][1] = i
            j = j + 1
        i = 0
        area0 = area[0][0]
        border0 = area[0][1]
        while area[i][0] > 0:
            i = i + 1
            if area[i][0] > area0:
                area0 = area[i][0]
                border0 = area[i][1]
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

# 候选框事件输出率
def event_output(rect,events):
    output=0
    for i in events:
        if (rect[0]<=i[2]<=rect[0]+rect[2] and rect[1]<=i[1]<=rect[1]+rect[3]):
            output+=1
    return (output/(rect[2]*rect[3]))

# 候选框事件长宽比
def length_width(rect,events):
    length=()
    width=()
    for i in events:
        if (rect[0]<=i[2]<=rect[0]+rect[2] and rect[1]<=i[1]<=rect[1]+rect[3]):
            length=np.append(length,i[1])
            width=np.append(width,i[2])
    if (np.max(width)-np.min(width))!=0:
        return (np.max(length)-np.min(length))/(np.max(width)-np.min(width))
    else:
        return 0

# 候选框内角点数
def corner_in_box(rect,img):
    dst = cv2.cornerHarris(src=img, blockSize=2, ksize=3, k=0.04)
    dst1 = dst.copy()
    dst1[dst <= 0.01 * dst.max()] = 0
    score_nms = corner_nms(dst1) 
    num=0
    for i in range(len(score_nms)):
        for j in range(len(score_nms[i])):
            if score_nms[i][j]!=0:
                if(rect[0]<=i<=rect[0]+rect[2] and rect[1]<=j<=rect[1]+rect[3]):
                    num+=1
    return num

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

    # build dataset
    data_loader_train = build_dataloader(args, partition='train') 
    data_loader_val   = build_dataloader(args, partition='test')
    
    # Train model
    for epoch in range(stat['epoch'], args.epochs):

        total_loss = 0
        for batch_idx, (samples, targets) in enumerate(data_loader_train):
            result = to_do_train(samples, targets)
            for i in range(len(result)):
                x_train=[]
                matrix_init=fill(samples[i]['events'])
                for n in range(len(result[i])):
                    # eventoutput=event_output(result[i][n],samples[i]['events'])
                    # num_corner=corner_in_box(result[i][n],matrix_init.astype(np.uint8))
                    # len_wid=length_width(result[i][n],samples[i]['events'])
                    x_train.append([corner_in_box(result[i][n],matrix_init.astype(np.uint8)),event_output(result[i][n],samples[i]['events']),length_width(result[i][n],samples[i]['events'])])
                breakpoint()
