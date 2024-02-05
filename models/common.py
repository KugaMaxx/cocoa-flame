import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_ops import pointnet2_utils


def get_activation(activation):
    if activation.lower() == 'relu':
        return F.relu
    if activation.lower() == 'gelu':
        return F.gelu
    if activation.lower() == 'rrelu':
        return F.rrelu
    if activation.lower() == 'selu':
        return F.selu
    if activation.lower() == 'silu':
        return F.silu
    if activation.lower() == 'leaky_relu':
        return F.leaky_relu
    raise RuntimeError(F"Unsupported activation function: '{activation}'.")


def get_points_by_index(points, idx):
    device = points.device
    batch_size, *_  = points.shape
    
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(batch_size, dtype=torch.long).to(device)
    batch_indices = batch_indices.view(view_shape).repeat(repeat_shape)
    
    return points[batch_indices, idx, :]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=False, 
                 activation="relu", residual_factor=1.0, **kwargs):
        super().__init__()
        assert in_channels == out_channels, \
            f"The in_channels ({in_channels}) is not equal to out_channels ({out_channels})."
        
        mid_channels = int(in_channels * residual_factor)

        self.activation = get_activation(activation)

        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size, bias=bias)
        self.conv2 = nn.Conv1d(mid_channels, out_channels, kernel_size, bias=bias)
        
        self.norm1 = nn.BatchNorm1d(mid_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out + x)

        return out
    

class LocalGrouper(nn.Module):
    def __init__(self, in_channels, num_samples, k_neighbors=32, use_xyz=False, 
                 normalize="center", **kwargs):
        super().__init__()
        assert normalize in ["center", "anchor"] or normalize is None, \
            f"Unrecognized normalize type: '{normalize}'."

        self.num_samples = num_samples
        self.k_neighbors = k_neighbors

        self.use_xyz = use_xyz
        self.normalize = normalize

        if self.normalize is not None:
            add_channels = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, in_channels + add_channels]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, in_channels + add_channels]))

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor):
        xyz = xyz.contiguous()
        batch_size, num_points, xyz_dims = xyz.shape

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.num_samples).long()
        fps_xyz = get_points_by_index(xyz, fps_idx)  # [batch_size, num_samples, xyz_dims]
        fps_feat = get_points_by_index(feat, fps_idx)  # [batch_size, num_samples, feat_dims]

        # get neighbors of farthest points by distance
        dist = torch.cdist(fps_xyz, xyz, p=2)
        _, group_idx = torch.topk(dist, self.k_neighbors, dim=-1, largest=False, sorted=False)
        grouped_xyz = get_points_by_index(xyz, group_idx)  # [batch_size, num_samples, k_neighbors, xyz_dims]
        grouped_feat = get_points_by_index(feat, group_idx)  # [batch_size, num_samples, k_neighbors, feat_dims]

        # concatenate if use xyz [batch_size, num_samples, k_neighbors, xyz_dims + feat_dims]
        if self.use_xyz:
            grouped_feat = torch.cat([grouped_feat, grouped_xyz], dim=-1)

        # normalize if not none [batch_size, num_samples, 1, xyz_dims + feat_dims]
        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_feat, dim=2, keepdim=True)
            if self.normalize =="anchor":
                mean = torch.cat([fps_feat, fps_xyz], dim=-1) if self.use_xyz else fps_feat
                mean = mean.unsqueeze(dim=-2)
            
            std = torch.std((grouped_feat - mean).view(batch_size, -1), dim=-1).view(batch_size, 1, 1, 1)
            grouped_feat = (grouped_feat - mean) / (std + 1e-5)
            grouped_feat = self.affine_alpha * grouped_feat + self.affine_beta

        # concatenate
        fps_feat = torch.cat([grouped_feat, 
                              fps_feat.unsqueeze(2).repeat(1, 1, self.k_neighbors, 1)], dim=-1)

        return fps_xyz, fps_feat


if __name__ == '__main__':
    # set random seed
    torch.manual_seed(0)

    # test residual block
    batch_size, in_channels, num_points = 1, 3, 16
    residual_block = ResidualBlock(in_channels, in_channels)
    input_tensor   = torch.randn(batch_size, in_channels, num_points)
    output_tensor  = residual_block(input_tensor)

    print("Input:",  input_tensor)
    print("Output:", output_tensor)

    # test local grouper
    batch_size, num_points, xyz_dims, feat_dims = 2, 16, 3, 64
    xyz  = torch.randn(batch_size, num_points, xyz_dims).to("cuda")
    feat = torch.randn(batch_size, num_points, feat_dims).to("cuda")
    local_grouper = LocalGrouper(feat_dims, 10, k_neighbors=8, use_xyz=True).to("cuda")

    print("Feat: ", local_grouper(xyz, feat)[0])
