import numpy as np
import torch


# before: 
def IoU(box1, box2) -> float:
    weight = max(min(box1[2], box2[2]) - max(box1[0], box2[0]), 0)
    height = max(min(box1[3], box2[3]) - max(box1[1], box2[1]), 0)
    s_inter = weight * height
    s_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    s_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    s_union = s_box1 + s_box2 - s_inter
    return s_inter / s_union


def transform(box) -> float:
    box1=[box[0],box[1]-box[3],box[0]+box[2],box[1]]
    return box1


# after: 
def calc_iou(boxA, boxB):
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
        return 0.0  # 框不相交，IOU为0

    # 计算交集部分面积
    interArea = max(0, x_inter2 - x_inter1) * max(0, y_inter2 - y_inter1)

    # 分别计算两个box的面积
    area_box1 = (rb_x1 - tl_x1) * (rb_y1 - tl_y1)
    area_box2 = (rb_x2 - tl_x2) * (rb_y2 - tl_y2)

    #计算IOU，交集比并集，并集面积=两个矩形框面积和-交集面积
    return interArea / (area_box1 + area_box2 - interArea)


if __name__ == '__main__':
    # test hungarian matcher
    num_labels, num_queries, num_classes = [2, 4, 4], 100, 2

    targets = list()
    for i, num_label in enumerate(num_labels):
        targets.append({
            'labels': torch.randint(0, num_classes, [num_label]),
            'boxes': torch.rand([num_label, 4]),
            'resolution': (346, 260)
        })

    batch_size = len(num_labels)
    outputs = {
        "pred_logits": torch.rand([batch_size, num_queries, num_classes]).softmax(-1),
        "pred_boxes": torch.rand([batch_size, num_queries, 4])
    }

    # print(targets)
    # print(outputs)
    
    # before: 字典直接用 dict[key] 的形式访问即可
    pred_boxes = []
    first_boxes = []
    for x, y in outputs.items():
        pred_boxes.append(y.numpy())
    first_boxes = pred_boxes[1]
    
    # after: 修改如下
    first_boxes = outputs['pred_boxes'].numpy()


    # before: python 在这样的 for 循环中会有计算冗余
    target_box = []
    for i in targets:
        target_box.append(i["boxes"].numpy())

    # after: 循环写在 [] 内部会提高运行效率
    target_box = [tar["boxes"].numpy() for tar in targets]


    # before: 设计了两个大的 for 循环，第一次判断是否相交，第二次计算iou
    # print(len(firstboxes[0]))
    i = 0
    result = np.zeros((3,100))
    for target in target_box:   # 可以用 for i, target in enumerate(target_box)
        i = i + 1
        for m in target:
            minx1, miny1, maxx1, maxy1 = (m[0], m[1]-m[3], m[0]+m[2], m[1])
            j = 0
            for box in first_boxes[i-1]:
                j = j + 1
                minx2, miny2, maxx2, maxy2= (box[0], box[1]-box[3], box[0]+box[2], box[1])
                minx = max(minx1, minx2)
                miny = max(miny1, miny2)
                maxx = min(maxx1, maxx2)
                maxy = min(maxy1, maxy2)
                if minx < maxx and miny < maxy:
                    result[i-1][j-1]=1
    print(result)
    iou = []
    for i in range(len(target_box)):
        for j in range(len(first_boxes[0])):
            if(result[i][j]):
                for k in range(len(target_box[i])):
                    iou.append({
                            'i':i,
                            'j':j,
                            'k':k,
                            'iou_ijk':IoU(transform(target_box[i][k]),transform(first_boxes[i][j]))
                    })
    print(iou)

    # after: 在 python 中，多考虑如何在更少的 for 循环中实现目标
    results = []
    for i, tgt_box in enumerate(target_box):
        for target in tgt_box:
            results.append(
                np.array([calc_iou(target, output) for output in first_boxes[i]])    # 取对应 batch 的 first_boxes
            )


    # 终极版，使用 torch 自带的方法，挑战最少 for 循环、最高运算效率和最简代码
    xywh_to_xyxy = lambda x: torch.stack([x[:, 0], x[:, 1], x[:, 0] + x[:, 2], x[:, 1] + x[:, 3]], dim=-1)
    
    tgt_boxes = [xywh_to_xyxy(tar["boxes"]) for tar in targets]
    pre_boxes = [xywh_to_xyxy(pre) for pre in outputs['pred_boxes']]

    from torchvision.ops import box_iou
    results = [box_iou(tgt, pre) for tgt, pre in zip(tgt_boxes, pre_boxes)]
    
    # x1, y1, w1, h1 = 0, 0, 3, 3
    # x2, y2, w2, h2 = 1, 1, 1, 1
    # print(calc_iou([x1, y1, w1, h1], [x2, y2, w2, h2]))
    # print(box_iou(xywh_to_xyxy(torch.tensor([[x1, y1, w1, h1]])), 
    #               xywh_to_xyxy(torch.tensor([[x2, y2, w2, h2]]))))
