import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_ops import pointnet2_utils
from common import MLP, ResidualBlock, LocalGrouper
from common import get_activation, get_points_by_index


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
                 num_layers=4, num_classes=4, num_queries=100, **kwargs):
        super().__init__()
        self.encrypt_dim = encrypt_dim
        self.encrypt_pts = encrypt_pts
        self.num_layers  = num_layers

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

        # classification embedding
        # self.class_embed = nn.Linear(last_channels, num_classes + 1)
        self.class_embed = MLP(last_channels, last_channels, num_classes + 1, num_layers=3)

        # bounding box embedding
        self.bbox_embed = MLP(last_channels, last_channels, 4, num_layers=3)

    def forward(self, x):
        # sample input tensor
        xyz = x.permute(0, 2, 1).contiguous()
        idx = pointnet2_utils.furthest_point_sample(xyz, self.encrypt_pts).long()
        xyz = get_points_by_index(xyz, idx)  # [batch_size, encrypt_pts, 3]

        # embedding
        feat = self.coord_encrypt(xyz.permute(0, 2, 1))  # [batch_size, encrypt_dim, encrypt_pts]

        # backbone
        for layer in self.backbone:
            xyz, feat = layer(xyz, feat.permute(0, 2, 1))
        feat = feat.permute(0, 2, 1)

        # predictor
        outputs_class = F.softmax(self.class_embed(feat), dim=2)
        outputs_coord = self.bbox_embed(feat).sigmoid()

        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}


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
