import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNormRelu(nn.Module):
    def __init__(self, conv_type='1d', in_channels=3, out_channels=64, downsample=False,
                 kernel_size=None, stride=None, padding=None, norm='BN', num_groups=1, 
                 leaky=False):
        super().__init__()
        if kernel_size is None:
            if downsample:
                kernel_size, stride, padding = 4, 2, 1
            else:
                kernel_size, stride, padding = 3, 1, 1

        if conv_type == '2d':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            )
            if norm == 'BN':
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm == 'IN':
                self.norm = nn.InstanceNorm2d(out_channels)
            elif norm == 'GN':
                self.norm = nn.GroupNorm(num_groups, out_channels)
            else:
                raise NotImplementedError
        elif conv_type == '1d':
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            )
            if norm == 'BN':
                self.norm = nn.BatchNorm1d(out_channels)
            elif norm == 'IN':
                self.norm = nn.InstanceNorm1d(out_channels)
            elif norm == 'GN':
                self.norm = nn.GroupNorm(num_groups, out_channels)
            else:
                raise NotImplementedError
        nn.init.kaiming_normal_(self.conv.weight)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True) if leaky else nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if isinstance(self.norm, nn.InstanceNorm1d):
            x = self.norm(x.permute((0, 2, 1))).permute((0, 2, 1))  # normalize on [C]
        else:
            x = self.norm(x)
        x = self.act(x)
        return x

class Poses_Encoder(nn.Module):
    def __init__(self, n_dim):
        super().__init__()

        leaky = 0.1
        norm = 'BN'
        num_groups = 1
        out_channels = 32
        in_channels = n_dim
        
        self.blocks = nn.Sequential(
            ConvNormRelu('1d', in_channels, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, out_channels, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky),
        )

    def forward(self, x):
        x = x.reshape(list(x.shape[:2])+[-1]).permute([0, 2, 1])

        x = self.blocks(x)
        x = F.interpolate(x, 1).squeeze(-1)

        return x

class Poses_Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        leaky = 0.1
        norm = 'BN'
        num_groups = 1
        in_channels = 32
        
        self.d5 = ConvNormRelu('1d', in_channels, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)
        self.d4 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)
        self.d3 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)
        self.d2 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)
        self.d1 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)

        self.blocks = nn.Sequential(
            ConvNormRelu('1d', 256, 256, downsample=False, kernel_size=3, stride=1, padding=0, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            nn.Conv1d(256, 27, kernel_size=1, bias=True)
        )

    def forward(self, x):
        x = F.interpolate(x.unsqueeze(-1), 2)

        x = self.d5(F.interpolate(x, x.shape[-1]*2, mode='linear', align_corners=True))
        x = self.d4(F.interpolate(x, x.shape[-1]*2, mode='linear', align_corners=True))
        x = self.d3(F.interpolate(x, x.shape[-1]*2, mode='linear', align_corners=True))
        x = self.d2(F.interpolate(x, x.shape[-1]*2, mode='linear', align_corners=True))
        # x = self.d1(F.interpolate(x, x.shape[-1]*2, mode='linear', align_corners=True))

        x = self.blocks(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, n_pose, n_dim):
        super().__init__()

        self.encoder = Poses_Encoder(n_dim)
        self.decoder = Poses_Decoder()
        self.n_dim = n_dim

    def forward(self, x, num_frames):
        code = self.encoder(x)
        x = self.decoder(code)

        x = x.permute([0,2,1]).reshape(-1, num_frames, self.n_dim)
        return x


class VAE_Encoder(nn.Module):
    def __init__(self, n_dim):
        super().__init__()

        leaky = 0.1
        norm = 'BN'
        num_groups = 1
        out_channels = 32 * 2
        in_channels = n_dim
        
        self.blocks = nn.Sequential(
            ConvNormRelu('1d', in_channels, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, out_channels, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky),
        )

    def forward(self, x):
        x = x.reshape(list(x.shape[:2])+[-1]).permute([0, 2, 1])

        x = self.blocks(x)
        x = F.interpolate(x, 1).squeeze(-1)
        mu = x[:, 0::2]
        logvar = x[:, 1::2]

        return mu, logvar


class VAE(nn.Module):
    def __init__(self, n_pose, n_dim):
        super().__init__()

        self.encoder = VAE_Encoder(n_dim)
        self.decoder = Poses_Decoder()
        self.n_dim = n_dim

    def forward(self, x, num_frames):
        mu, logvar = self.encoder(x)
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        code =  mu + eps * std
        x = self.decoder(code)

        x = x.permute([0,2,1]).reshape(-1, num_frames, self.n_dim)
        return x, mu, logvar


class Poses_Encoder2(nn.Module):
    def __init__(self, n_dim):
        super().__init__()

        leaky = 0.1
        norm = 'BN'
        num_groups = 1
        out_channels = 32
        in_channels = n_dim
        
        self.blocks = nn.Sequential(
            ConvNormRelu('1d', in_channels, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky), # 50
            ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky), # 25
            ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky), # 12
            ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky), # 6
            ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky), # 3
            ConvNormRelu('1d', 256, out_channels, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky), # 1
        )

    def forward(self, x):
        x = x.reshape(list(x.shape[:2])+[-1]).permute([0, 2, 1])

        x = self.blocks(x)
        x = x.squeeze(-1)

        return x


class Poses_Decoder2(nn.Module):
    def __init__(self, n_dim, n_pose):
        super().__init__()

        leaky = 0.1
        norm = 'BN'
        num_groups = 1
        in_channels = 32

        self.blocks = nn.Sequential(
            ConvNormRelu('1d_rev', in_channels, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d_rev', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky), # 2
            ConvNormRelu('1d_rev', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky), # 4
            ConvNormRelu('1d_rev', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky), # 8
            ConvNormRelu('1d_rev', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky), # 16
            ConvNormRelu('1d_rev', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky), # 32
            ConvNormRelu('1d_rev', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky), # 64
            ConvNormRelu('1d_rev', 256, n_dim, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky), # 128
        )
        self.linear = nn.Linear(128, 100)


    def forward(self, x):
        x = x.unsqueeze(-1)

        x = self.blocks(x) # [128, dim, 128]
        x = self.linear(x) # [128, dim, 100]
        return x


class Trinity_Autoencoder(nn.Module):
    def __init__(self, n_pose, n_dim):
        super().__init__()

        self.encoder = Poses_Encoder2(n_dim)
        self.decoder = Poses_Decoder2(n_dim, n_pose)
        self.n_dim = n_dim

    def forward(self, x, num_frames):
        code = self.encoder(x)
        x = self.decoder(code)

        x = x.permute([0,2,1]).reshape(-1, num_frames, self.n_dim)
        return x