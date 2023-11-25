import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, groups=1, expansion=1.0):
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=int(in_channels * expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(in_channels * expansion)),
            nn.ReLU(inplace=True)
        )
        
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(in_channels * expansion), out_channels=out_channels,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(in_channels * expansion), out_channels=out_channels,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(out_channels)
            )
        
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class Extraction(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, groups=1, expansion=1.0, 
                 pre_block=1, pos_block=1, use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super().__init__()

        # Embedding
        in_channels = 3 + 2 * in_channels if use_xyz else 2 * in_channels
        self.transfer = EmbedBlock(in_channels, out_channels)
        
        # Pre Extraction
        self.pre_extraction = nn.ModuleList()
        for _ in range(pre_block):
            self.pre_extraction.append(
                ResBlock(in_channels=out_channels, out_channels=out_channels, 
                         kernel_size=kernel_size, bias=bias, groups=groups, expansion=expansion)
            )

        # Pos Extraction
        self.pos_extraction = nn.ModuleList()
        for _ in range(pos_block):
            self.pos_extraction.append(
                ResBlock(in_channels=out_channels, out_channels=out_channels, 
                         kernel_size=kernel_size, bias=bias, groups=groups, expansion=expansion)
            )

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.pre_extraction(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        x = self.pos_extraction(x)  # [b, d, k]
        return x


class Model(nn.Module):
    def __init__(self, point_num=49, class_num=2, embed_dim=64, 
                 kernel_size=1, bias=True, groups=1, expansion=1.0, 
                 pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2], use_xyz=True):
        super().__init__()

        # Detect the number of layers
        self.layer_num = len(pre_blocks)
        assert self.layer_num == len(pre_blocks) == len(pos_blocks)

        last_channels = embed_dim

        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=last_channels, kernel_size=1, bias=True),
            nn.BatchNorm1d(last_channels),
            nn.ReLU(inplace=True)
        )

        self.extractor = nn.ModuleList()
        for pre_block, pos_block in zip(pre_blocks, pos_blocks):
            out_channels = last_channels // TODO:
            self.extractor.append(Extraction(in_channels=1, out_channels=1, kernel_size=kernel_size, bias=bias, groups=groups, expansion=expansion, pre_blocks=2, pos_blocks=2, use_xyz=use_xyz))


        self.classifier = nn.Sequential(
            nn.Linear(last_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, class_num)
        )

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()

        x = self.embedding(x)  # B,D,N
        for i in range(self.layer_num):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            # xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
            # x = self.pre_blocks_list[i](x)  # [b,d,g]
            # x = self.pos_blocks_list[i](x)  # [b,d,g]
            x = self.extractor[i](x)
        
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        x = self.classifier(x)
        return x

