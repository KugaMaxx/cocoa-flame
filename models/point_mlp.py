
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from pointnet2_ops import pointnet2_utils


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'leakyrelu0.2':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
        return nn.ReLU(inplace=True)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    dists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(dists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channels, kernel_size=1, bias=True, activation='relu', groups=1, res_expansion=1.0):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=int(channels * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channels * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channels * res_expansion), out_channels=channels,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channels),
                self.act,
                nn.Conv1d(in_channels=channels, out_channels=channels,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channels),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channels * res_expansion), out_channels=channels,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channels)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class LocalGrouper(nn.Module):
    def __init__(self, channels, sample_num, neighbor_num, use_xyz=True, normalize="anchor"):
        super().__init__()
        self.sample_num = sample_num
        self.neighbor_num = neighbor_num
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channels + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channels + add_channel]))

    def forward(self, point_xyz, point_feat):
        """
        :param point_xyz: [batch, point, 3]
        :param point_feat: [batch, point, dim]
        """
        S = self.sample_num
        B, N, C = point_xyz.shape
        point_xyz = point_xyz.contiguous()  # xyz [batch, point, 3]

        # Sample points and related features by furthest sampling
        fps_idx = pointnet2_utils.furthest_point_sample(point_xyz, self.sample_num).long()  # [batch, sample]
        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.sample_num, replacement=False).long()
        # fps_idx = farthest_point_sample(point_xyz, self.sample_num).long()
        sampled_xyz = index_points(point_xyz, fps_idx)  # [batch, sample, 3]
        sampled_feat = index_points(point_feat, fps_idx)  # [batch, sample, dim]

        # Search and group neighbor points around sample points
        idx = knn_point(self.neighbor_num, point_xyz, sampled_xyz)  # [batch, sample, neighbor]
        # idx = query_ball_point(radius, nsample, point_xyz, sampled_xyz)
        grouped_xyz = index_points(point_xyz, idx)  # [batch, sample, neighbor, 3]
        grouped_feat = index_points(point_feat, idx)  # [batch, sample, neighbor, dim]

        # Concatenate xyz in the end
        if self.use_xyz:
            grouped_feat = torch.cat([grouped_feat, grouped_xyz], dim=-1)  # [batch, sample, neighbor, dim + 3]

        # Normalize
        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_feat, dim=2, keepdim=True)
            if self.normalize =="anchor":
                mean = torch.cat([sampled_feat, sampled_xyz], dim=-1) if self.use_xyz else sampled_feat
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]	
            std = torch.std((grouped_feat - mean).reshape(B,-1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_feat = (grouped_feat - mean) / (std + 1e-5)
            grouped_feat = self.affine_alpha * grouped_feat + self.affine_beta

        # Unsqueeze sampled features and concatenate behind grouped features
        sampled_feat = sampled_feat.view(B, S, 1, -1).repeat(1, 1, self.neighbor_num, 1)  # [batch, sample, neighbor, dim]
        sampled_feat = torch.cat([grouped_feat, sampled_feat], dim=-1)  # [batch, sample, neighbor, 2 * dim (+ 3)]
        return sampled_xyz, sampled_feat


class PreExtraction(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, activation='relu', use_xyz=True, blocks=1, groups=1, res_expansion=1):
        super().__init__()
        in_channels = 2 * in_channels + 3 if use_xyz else 2 * in_channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, bias=bias, activation=activation,
                                groups=groups, res_expansion=res_expansion)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        """
        :param x: [batch, sample, neighbor, 2 * dim (+ 3)]
        """
        b, s, n, d = x.size()  # [batch, sample, neighbor, 2 * dim (+ 3)]
        x = x.permute(0, 1, 3, 2)  # [batch, sample, 2 * dim (+ 3), neighbor]
        x = x.reshape(-1, d, n)  # [batch * sample, 2 * dim (+ 3), neighbor]

        # Embed and enlarge dims
        x = self.transfer(x)  # [batch * sample, dim, neighbor]
        batch_size, _, _ = x.size()

        # Extract features
        x = self.operation(x)  # [batch * sample, dim, neighbor]

        # Adaptive max pooling
        x = F.adaptive_max_pool1d(x, 1)  # [batch * sample, dim, 1]
        x = x.view(batch_size, -1)  # [batch * sample, dim]
        
        # back to original batch
        x = x.reshape(b, s, -1)  # [batch, sample, dim]
        x = x.permute(0, 2, 1)  # [batch, dim, sample]

        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, bias=True, activation='relu', use_xyz=True, blocks=1, groups=1, res_expansion=1):
        super().__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, bias=bias, activation=activation,
                                groups=groups, res_expansion=res_expansion)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        """
        :param x: [batch, dim, sample]
        """
        return self.operation(x)  # [batch, dim, sample]
    

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, sample_num, neighbor_num, 
                 pre_blocks=1, pos_blocks=1, use_xyz=True, normalize="anchor",
                 groups=1, res_expansion=1, bias=True, activation='relu'):
        super().__init__()
        # 
        self.local_grouper = LocalGrouper(in_channels, sample_num=sample_num, neighbor_num=neighbor_num, 
                                          use_xyz=use_xyz, normalize=normalize)

        #
        self.pre_extractor = PreExtraction(in_channels, out_channels, blocks=pre_blocks, groups=groups, use_xyz=use_xyz,
                                           res_expansion=res_expansion, bias=bias, activation=activation)
        
        #
        self.pos_extractor = PosExtraction(out_channels, blocks=pos_blocks, groups=groups, use_xyz=use_xyz,
                                           res_expansion=res_expansion, bias=bias, activation=activation)

    def forward(self, point_xyz, point_feat):
        # input:  point_xyz[batch, point, 3],  x[batch, point, in_dim]
        # output: point_xyz[batch, sample, 3], x[batch, sample, neighbor, 2 * in_dim (+ 3)]
        point_xyz, point_feat = self.local_grouper(point_xyz, point_feat.permute(0, 2, 1))

        # input:  x[batch, sample, neighbor, 2 * in_dim (+ 3)]
        # output: x[batch, out_dim, sample]
        point_feat = self.pre_extractor(point_feat)

        # input:  feat[batch, out_dim, sample]
        # output: feat[batch, out_dim, sample]
        point_feat = self.pos_extractor(point_feat)

        return point_xyz, point_feat


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, de_blocks=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super().__init__()
        self.fuse = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        self.extraction = PosExtraction(out_channels, blocks=de_blocks, groups=groups,
                                        res_expansion=res_expansion, bias=bias, activation=activation)

    def forward(self, point_xyz1, point_xyz2, point_feat1, point_feat2):
        """
        Input:
            point_xyz1: input points position data, [B, N, 3]
            point_xyz2: sampled input points position data, [B, S, 3]
            point_feat1: input points data, [B, D', N]
            point_feat2: input points data, [B, D'', S]
        Return:
            new_point_feat: upsampled points data, [B, D''', N]
        """
        # Initialize
        B, N, C = point_xyz1.shape  # [batch, point, 3]
        _, S, _ = point_xyz2.shape  # [batch, sample, 3]

        # Distance interpolation
        point_feat2 = point_feat2.permute(0, 2, 1)  # [batch, sample, en_dim]
        if S == 1:
            interpolated_feat = point_feat2.repeat(1, N, 1)
        else:
            dists = square_distance(point_xyz1, point_xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feat = torch.sum(index_points(point_feat2, idx) * weight.view(B, N, 3, 1), dim=2)

        # Interpolate and concatenate features
        if point_feat1 is not None:
            point_feat1 = point_feat1.permute(0, 2, 1)  # [batch, point, de_dim]
            new_point_feat = torch.cat([point_feat1, interpolated_feat], dim=-1)
        else:
            new_point_feat = interpolated_feat

        # Extract features
        new_point_feat = new_point_feat.permute(0, 2, 1)  # [batch, de_dim + en_dim, point]
        new_point_feat = self.fuse(new_point_feat)
        new_point_feat = self.extraction(new_point_feat)  # [batch, de_dim, point]
        
        return new_point_feat


class PointMLP(nn.Module):
    def __init__(self, class_num=50, embed_dim=64, gmp_dim=64, cls_dim=64, use_xyz=True,
                 groups=1, res_expansion=1.0, bias=True, activation="relu", 
                 sample_nums=[512, 128, 32, 8], neighbor_nums=[32, 32, 32, 32], normalize="anchor",
                 en_dims=[128, 256, 512, 1024], de_dims=[512, 256, 128, 128], 
                 pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2], de_blocks=[4, 4, 4, 4], **kwargs):
        super(PointMLP, self).__init__()
        
        # Check arguments
        assert len(pre_blocks) == len(pos_blocks) == len(neighbor_nums), \
            "Please check stage number consistent for pre_blocks, pos_blocks neighbor_nums, reducers."
        assert len(en_dims) == len(de_dims) == len(de_blocks), \
            "Please check stage number consistent for en_dims, de_dims, en_blocks, de_blocks"

        # Embedding features
        self.embedding = ConvBNReLU1D(6, embed_dim, bias=bias, activation=activation)

        # Initialize basic info
        last_channel = embed_dim

        # Building Encoder
        en_dims.insert(0, embed_dim)
        self.encoder_list = nn.ModuleList()
        for i in range(len(pre_blocks)):
            # update channels
            out_channel = en_dims[i + 1]

            self.encoder_list.append(
                Encoder(last_channel, out_channel, sample_num=sample_nums[i], neighbor_num=neighbor_nums[i], 
                        pre_blocks=pre_blocks[i], pos_blocks=pos_blocks[i], use_xyz=use_xyz, normalize=normalize, 
                        groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )

            # update channels
            last_channel = out_channel

        # Reverse
        en_dims.reverse()

        # Building Decoder
        de_dims.insert(0, en_dims[0])
        self.decoder_list = nn.ModuleList()
        for i in range(len(de_blocks)):
            # update channels
            in_channel = last_channel + en_dims[i + 1]
            out_channel = de_dims[i + 1]

            self.decoder_list.append(
                Decoder(in_channel, out_channel, de_blocks=de_blocks[i], 
                        groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )

            # update channels
            last_channel = out_channel

        # Class label mapping
        self.cls_map = nn.Sequential(
            ConvBNReLU1D(16, cls_dim, bias=bias, activation=activation),
            ConvBNReLU1D(cls_dim, cls_dim, bias=bias, activation=activation)
        )

        # Global max pooling mapping
        self.gmp_map_list = nn.ModuleList()
        for en_dim in en_dims:
            self.gmp_map_list.append(ConvBNReLU1D(en_dim, gmp_dim, bias=bias, activation=activation))
        self.gmp_map_end = ConvBNReLU1D(gmp_dim*len(en_dims), gmp_dim, bias=bias, activation=activation)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv1d(gmp_dim + cls_dim + de_dims[-1], 128, 1, bias=bias),
            nn.BatchNorm1d(128),
            nn.Dropout(),
            nn.Conv1d(128, class_num, 1, bias=bias)
        )

    def forward(self, x, norm_plt, cls_label):
        """
        :param x: [batch, 3, point]
        """
        # Embedding points
        xyz = x.permute(0, 2, 1)  # [batch, point, 3]
        x = torch.cat([x, norm_plt], dim=1)  # [batch, 6, point]
        x = self.embedding(x)  # [batch, dim, point]
        
        # Initialize skip connection
        x_list, xyz_list = [x], [xyz]

        # Run encoder
        for i in range(len(self.encoder_list)):
            xyz, x = self.encoder_list[i](xyz, x)
            
            # store output
            xyz_list.append(xyz)
            x_list.append(x)

        # Reverse
        xyz_list.reverse()
        x_list.reverse()

        # Run decoder
        for i in range(len(self.decoder_list)):
            x = self.decoder_list[i](xyz_list[i + 1], xyz_list[i], x_list[i + 1], x)

        # Require the global context
        gmp_list = []
        for i in range(len(x_list)):
            gmp_list.append(F.adaptive_max_pool1d(self.gmp_map_list[i](x_list[i]), 1))
        global_context = self.gmp_map_end(torch.cat(gmp_list, dim=1)) # [b, gmp_dim, 1]

        # Here is the cls_token
        cls_token = self.cls_map(cls_label.unsqueeze(dim=-1))  # [b, cls_dim, 1]
        x = torch.cat([x, global_context.repeat([1, 1, x.shape[-1]]), cls_token.repeat([1, 1, x.shape[-1]])], dim=1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        
        return x


def pointMLP(class_num=50, **kwargs) -> PointMLP:
    return PointMLP(class_num=class_num, embed_dim=64, gmp_dim=64, cls_dim=64, use_xyz=True,
                    groups=1, res_expansion=1.0, bias=True, activation="relu", 
                    sample_nums=[512, 128, 32, 8], neighbor_nums=[32, 32, 32, 32], normalize="anchor",
                    en_dims=[128, 256, 512, 1024], de_dims=[512, 256, 128, 128],
                    pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2], de_blocks=[4, 4, 4, 4], **kwargs)


if __name__ == '__main__':
    torch.manual_seed(0)
    data = torch.rand(2, 3, 2048).to('cuda')
    norm = torch.rand(2, 3, 2048).to('cuda')
    cls_label = torch.rand([2, 16]).to('cuda')
    print("===> testing modelD ...")
    model = pointMLP(50).to('cuda')
    out = model(data, norm, cls_label)  # [2,2048,50]
    print(out)
