
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch import einsum
# from einops import rearrange, repeat


# from pointnet2_ops import pointnet2_utils


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
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouper(nn.Module):
    def __init__(self, in_channels, points_num, k_neighbor, use_xyz=True, normalize="center", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param points_num: furthest points number
        :param k_neighbor: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.points_num = points_num
        self.k_neighbor = k_neighbor
        self.use_xyz = use_xyz
        self.normalize = normalize.lower() if normalize is not None else None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channels = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, in_channels + add_channels]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, in_channels + add_channels]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.points_num
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.points_num, replacement=False).long()
        fps_idx = farthest_point_sample(xyz, self.points_num).long()
        # fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.points_num).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]

        idx = knn_point(self.k_neighbor, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize =="anchor":
                mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            std = torch.std((grouped_points-mean).reshape(B,-1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points-mean)/(std + 1e-5)
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.k_neighbor, 1)], dim=-1)
        return new_xyz, new_points


class EmbedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(EmbedBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, groups=1, res_expansion=1.0):
        super(ResBlock, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=int(in_channels * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(in_channels * res_expansion)),
            nn.ReLU(inplace=True)
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(in_channels * res_expansion), out_channels=in_channels,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(in_channels),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(in_channels * res_expansion), out_channels=out_channels,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(in_channels)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class Extraction(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, groups=1, res_expansion=1.0, 
                 pre_block=1, pos_block=1, use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        """
        super(Extraction, self).__init__()
        
        # 
        in_channels = 3 + 2 * in_channels if use_xyz else 2 * in_channels
        self.transfer = EmbedBlock(in_channels, out_channels, bias=bias)
        
        #
        pre_extractor = []
        for _ in range(pre_block):
            pre_extractor.append(
                ResBlock(in_channels=out_channels, out_channels=out_channels, 
                         groups=groups, res_expansion=res_expansion, bias=bias)
            )
        self.pre_extractor = nn.Sequential(*pre_extractor)

        #
        pos_extractor = []
        for _ in range(pos_block):
            pos_extractor.append(
                ResBlock(in_channels=out_channels, out_channels=out_channels, 
                         groups=groups, res_expansion=res_expansion, bias=bias)
            )
        self.pos_extractor = nn.Sequential(*pos_extractor)

    def forward(self, x):
        # 
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        
        #
        batch_size, _, _ = x.size()
        x = self.pre_extractor(x)  # [b, d, k]
        
        # 
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        
        #
        x = self.pos_extractor(x)  # [b, d, g]
        
        return x
    

class Model(nn.Module):
    def __init__(self, points=1024, candidate_num=128, class_num=40, embed_dim=64, kernel_size=1, bias=True, groups=1, 
                 res_expansions=[1, 1, 1, 1], dim_expansions=[2, 2, 2, 2],
                 pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2], use_xyz=True,
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], normalize="center", **kwargs):
        super(Model, self).__init__()

        self.layer_num = len(pre_blocks)

        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansions) == len(res_expansions), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."

        # initialize embedding
        last_channels = embed_dim
        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=last_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(last_channels),
            nn.ReLU(inplace=True)
        )

        self.grouper = nn.ModuleList()
        self.extractor = nn.ModuleList()

        anchor_points = points
        for i in range(len(pre_blocks)):
            in_channels  = last_channels
            out_channels = last_channels * dim_expansions[i]
            
            # append local_grouper_list
            k_neighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            self.grouper.append(
                LocalGrouper(in_channels, points_num=anchor_points, k_neighbor=k_neighbor, use_xyz=use_xyz, normalize=normalize)  # [b,g,k,d]
            )
            
            # append pre_block_list
            self.extractor.append(
                Extraction(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, bias=bias, groups=groups, res_expansion=res_expansions[i],
                           pre_block=pre_blocks[i], pos_block=pos_blocks[i], use_xyz=use_xyz)
            )

            last_channels = out_channels

        # fully connected layer
        self.locator = nn.Sequential(
            nn.Linear(last_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, candidate_num * 4)
        )

        self.classifier = nn.Sequential(
            nn.Linear(last_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, candidate_num * class_num)
        )

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        
        #
        x = self.embedding(x)  # B,D,N

        #
        for i in range(self.layer_num):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            xyz, x = self.grouper[i](xyz, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
            x = self.extractor[i](x)  # [b,d,g]

        #
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        
        #
        x = self.classifier(x)

        # locate = self.locator(x)
        # confidence = self.classifier(x)
        
        return x


def PointMLP(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=64, kernel_size=1, bias=False, groups=1, 
                 res_expansions=[1., 1., 1., 1.], dim_expansions=[2, 2, 2, 2], 
                 pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2], use_xyz=False,
                 k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], normalize="anchor", **kwargs)


def PointMLPElite(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=32, kernel_size=1, bias=False, groups=1, 
                 res_expansions=[.25, .25, .25, .25], dim_expansions=[2, 2, 2, 1], 
                 pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1], use_xyz=False,
                 k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], normalize="anchor", **kwargs)


if __name__ == '__main__':
    torch.manual_seed(0)
    data = torch.rand(2, 3, 1024).cuda()
    print("===> testing pointMLP ...")
    model = PointMLP().to('cuda')
    out = model(data)
    print(out)

