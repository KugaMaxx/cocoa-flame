import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_ops import pointnet2_utils
from scipy.optimize import linear_sum_assignment

from models.common import MLP, ResidualBlock, LocalGrouper
from models.common import get_activation, get_points_by_index
from utils.boxes import generalized_box_iou, box_cxcywh_to_xyxy


class ExtractionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 num_samples, k_neighbors=32, use_xyz=False,  # local grouper
                 kernel_size=1, bias=False, activation="relu", residual_factor=1.0,  # residual block
                 pre_block=2, pos_block=2, **kwargs):
        super().__init__()
        # Geometric Affine
        # Concatenate sampled(farthest) points and their neighbors along dims(channels)
        self.local_grouper = LocalGrouper(in_channels, num_samples, k_neighbors, use_xyz)
        mid_channels = in_channels * 2 + 3 if use_xyz else in_channels * 2

        # Embedding
        self.transfer_embed = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels)
        )
        self.activation = get_activation(activation)
        mid_channels = out_channels

        # Pre Extraction
        self.pre_extractor = nn.Sequential(
            *[ResidualBlock(
                mid_channels, out_channels,
                kernel_size=kernel_size,
                bias=bias,
                activation=activation,
                residual_factor=residual_factor
            ) for _ in range(pre_block)]
        )

        # Pos Extraction
        self.pos_extractor = nn.Sequential(
            *[ResidualBlock(
                mid_channels, out_channels,
                kernel_size=kernel_size,
                bias=bias,
                activation=activation,
                residual_factor=residual_factor
            ) for _ in range(pos_block)]
        )

    def forward(self, xyz, feat):
        # sample points
        new_xyz, new_feat = self.local_grouper(xyz, feat)
        
        # convolve neighbors' coordinates at each dimension
        # input: [batch_size, num_samples, k_neighbors, feat_dims]
        # output: [batch_size * num_samples, feat_dims, k_neighbors]
        batch_size, num_samples, k_neighbors, feat_dims = new_feat.shape
        new_feat = new_feat.permute(0, 1, 3, 2)
        new_feat = new_feat.view(-1, feat_dims, k_neighbors)
        new_feat = self.activation(self.transfer_embed(new_feat))
        
        # extract features by local neighbors
        # input: [batch_size * num_samples, feat_dims, k_neighbors]
        # output: [batch_size * num_samples, feat_dims, k_neighbors]
        new_feat = self.pre_extractor(new_feat)

        # adaptive max pooling
        # input: [batch_size * num_samples, feat_dims, k_neighbors]
        # output: [batch_size, feat_dims, num_samples]
        new_feat = F.adaptive_max_pool1d(new_feat, 1)
        new_feat = new_feat.view(batch_size, num_samples, feat_dims)
        new_feat = new_feat.permute(0, 2, 1)

        # extract features by sampled points
        # input: [batch_size, feat_dims, num_samples]
        # output : [batch_size, feat_dims, num_samples]
        new_feat = self.pos_extractor(new_feat)

        return new_xyz, new_feat


class PointMLP(nn.Module):
    def __init__(self, encrypt_dim=64, dim_expansion=2, encrypt_pts=1024, pts_reducer=2, 
                 num_layers=4, num_classes=1, **kwargs):
        super().__init__()
        self.encrypt_dim = encrypt_dim
        self.encrypt_pts = encrypt_pts
        self.num_layers  = num_layers

        # Store device type
        self.dummy_param = nn.Parameter(torch.empty(0))

        # Encrypt coordinates
        self.coord_encrypt = nn.Sequential(
            nn.Conv1d(3, encrypt_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(encrypt_dim),
            nn.ReLU(),
        )
        last_channels = encrypt_dim
        last_samplers = encrypt_pts

        # Backbone
        self.backbone = nn.ModuleList()
        for i in range(self.num_layers):
            in_channels  = int(last_channels)
            out_channels = int(last_channels * dim_expansion)
            num_samples  = int(last_samplers / pts_reducer)

            self.backbone.append(
                ExtractionBlock(in_channels, out_channels, num_samples)
            )

            last_channels = out_channels
            last_samplers = num_samples
        
        # TODO 如果精度不高，可以试试对 last_channels 降维，num_samples 升维，结合 detr 和 pointMLP 的 decoder 部分
        # classification embedding
        self.class_embed = MLP(last_channels, last_channels, num_classes + 1, num_layers=3)

        # bounding box embedding
        self.bbox_embed = MLP(last_channels, last_channels, 4, num_layers=3)

    def forward(self, samples):
        # pre processing
        device = self.dummy_param.device
        xyz = torch.stack([self._pre_process(sample) for sample in samples]).to(device)

        # embedding
        xyz = xyz.contiguous()  # [batch_size, encrypt_pts, 3]
        feat = self.coord_encrypt(xyz.permute(0, 2, 1))  # [batch_size, encrypt_dim, encrypt_pts]

        # backbone
        for layer in self.backbone:
            xyz, feat = layer(xyz, feat.permute(0, 2, 1))
        feat = feat.permute(0, 2, 1)

        # predictor
        outputs_class = F.softmax(self.class_embed(feat), dim=2)
        outputs_coord = self.bbox_embed(feat).sigmoid()

        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
    
    @torch.no_grad()
    def _pre_process(self, sample):
        if sample['events'] is None:
            points = torch.zeros((self.encrypt_pts, 3))
            return points
        
        # convert sample to model input
        points = sample['events'][..., :3]
        delta_timestamp = points[..., 0] - points[..., 0][0]

        # set timestamp start from zero
        points = points.float()
        points[..., 0] = delta_timestamp * 1E-3

        # sampling points to predefined number
        num_points = len(points)
        if num_points > self.encrypt_pts:
            idx = torch.linspace(0, len(points) - 1, self.encrypt_pts).long()
            points = points[idx]
            return points
        else:
            points = torch.cat([points, torch.zeros((self.encrypt_pts - num_points, 3))])
            return points


class DETRLoss(nn.Module):
    def __init__(self, coef_class=1.0, coef_bbox=1.0, coef_giou=1.0, num_classes=1, eos_coef=0.01):
        super().__init__()
        self.coef_class  = coef_class
        self.coef_bbox   = coef_bbox
        self.coef_giou   = coef_giou
        self.num_classes = num_classes

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def forward(self, outputs, targets):
        # align type of device between outputs and targets
        device = next(iter(outputs.values())).device
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # run Hungarian matcher
        indices   = self._hungarian_matcher(outputs, targets)
        batch_idx = torch.cat([torch.full([len(tgt)], i) for i, (out, tgt) in enumerate(indices)])
        out_idx   = torch.cat([torch.tensor(out) for (out, _) in indices])
        tgt_idx   = torch.cat([torch.tensor(tgt) for (_, tgt) in indices])

        # calculate classification loss
        out_logits = outputs['pred_logits']
        tgt_labels = torch.full(out_logits.shape[:2], self.num_classes, dtype=torch.int64, device=device)
        tgt_labels[batch_idx, out_idx] = torch.cat([tgt['labels'][i] for tgt, (_, i) in zip(targets, indices)], dim=0)
        loss_class = F.cross_entropy(out_logits.transpose(1, 2), tgt_labels, self.empty_weight.to(device))
        loss_class = self.coef_class * loss_class.sum()

        # calculate bounding box loss
        out_boxes = outputs['pred_boxes'][batch_idx, out_idx]
        tgt_boxes = torch.cat([tgt['boxes'][i] for tgt, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(out_boxes, tgt_boxes, reduction='none')
        loss_bbox = self.coef_bbox * loss_bbox.sum()

        # calculate generalized iou loss
        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(out_boxes), 
                                                       box_cxcywh_to_xyxy(tgt_boxes)))
        loss_giou = self.coef_giou * loss_giou.sum()

        # final loss
        cost = loss_class + loss_bbox + loss_giou

        return cost

    @torch.no_grad()
    def _hungarian_matcher(self, outputs, targets):
        batch_size, num_queries, num_class = outputs['pred_logits'].shape

        out_prob = outputs['pred_logits'].flatten(0, 1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs['pred_boxes'].flatten(0, 1)   # [batch_size * num_queries, 4]

        tgt_ids  = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # compute the classification cost, 1 - probability, constant can be ommitted
        cost_class = -out_prob[:, tgt_ids]
        cost_class = self.coef_class * cost_class 

        # compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_bbox = self.coef_bbox * cost_bbox

        # compute giou cost between boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_giou = self.coef_giou + cost_giou

        # final cost matrix
        cost = cost_class + cost_bbox + cost_giou
        cost = cost.view(batch_size, num_queries, -1).to("cpu")
        cost = cost.split([len(t["labels"]) for t in targets], -1)

        # run hungarian matcher
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost)]

        return indices


if __name__ == '__main__':
    # set basic parameters
    device = "cuda"
    torch.manual_seed(0)

    # test extraction block
    batch_size, in_channels, out_channels = 2, 64, 128
    num_points, xyz_dims, feat_dims = 1024, 3, in_channels
    
    xyz  = torch.randn(batch_size, num_points, xyz_dims).to(device)
    feat = torch.randn(batch_size, num_points, feat_dims).to(device)
    extraction_block = ExtractionBlock(in_channels, out_channels, num_samples=512, k_neighbors=24).to(device)

    print("Feat: ", extraction_block(xyz, feat))

    # test point mlp
    point = torch.randn(batch_size, 3, num_points * 2).to(device)
    model = PointMLP().to(device)
    
    print("Output: ", model(point))

    # test hungarian matcher
    num_labels, num_queries, num_classes = [2, 4], 100, 92
    
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

    criterion = DETRLoss()
    criterion(outputs, targets)
