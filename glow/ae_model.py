import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNorm(nn.Module):
    def __init__(self, conv_type='1d', in_channels=3, out_channels=64, downsample=False,
                 kernel_size=None, stride=None, padding=None):
        
        super().__init__()
        if kernel_size is None:
            if downsample:
                kernel_size, stride = 4, 2
                if padding is None:
                    padding = 1
            else:
                kernel_size, stride = 3, 1
                if padding is None:
                    padding = 1
        
        if conv_type == '2d':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            )
        elif conv_type == '1d':
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            )
        
        self.norm = nn.BatchNorm1d(out_channels)
        self.actv = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.actv(x)
        return x


class ConvNormRes(nn.Module):
    def __init__(self, conv_type='1d', in_channels=3, out_channels=64, 
                 kernel_size=None, stride=None, padding=None):
        
        super().__init__()
        if kernel_size is None:
            kernel_size, stride = 3, 1
            if padding is None:
                padding = 1
        
        if conv_type == '2d':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            )
        elif conv_type == '1d':
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            )
        
        self.norm = nn.BatchNorm1d(out_channels)
        self.actv = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.norm(x)
        x += residual
        x = self.actv(x)
        return x


class Global_Encoder(nn.Module):
    def __init__(self, pose_dim, audio_dim):
        super().__init__()
        self.pose_encoder = TCN_Encoder(pose_dim)
        self.audio_encoder = TCN_Encoder(audio_dim)
        self.decode = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
        )


    def forward(self, x, y):
        x = self.pose_encoder(x) # [B, L, 64]
        y = self.audio_encoder(y) # [B, L, 64]
        out = torch.cat((x, y), dim=2) # [B, L, 128]
        out = self.decode(out) # [B, L, 32]

        return out # [B, L, 32]



class TCN_Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()

        out_channels = 64
        in_channels = dim
        
        self.cov_block = nn.Sequential(
            ConvNorm('1d', in_channels, 256, downsample=True),
            ConvNormRes('1d', 256, 256),
            ConvNormRes('1d', 256, 256),
            ConvNormRes('1d', 256, 256),
            ConvNormRes('1d', 256, 256),
            ConvNormRes('1d', 256, 256),
            ConvNorm('1d', 256, out_channels, downsample=True),
        )


    def forward(self, x):
        B, L , _, _ = x.shape
        x = x.reshape(B*L, x.shape[2], x.shape[3])
        x = x.permute([0, 2, 1])
        out = self.cov_block(x)
        out = F.interpolate(out, 1).squeeze(-1)
        out = out.reshape(B, L, 64)
        return out


class Global_VAE(nn.Module):
    def __init__(self, pose_dim, audio_dim):
        super().__init__()
        self.pose_encoder = TCN_Encoder(pose_dim)
        self.audio_encoder = TCN_Encoder(audio_dim)
        self.decode = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )


    def forward(self, x, y):
        x = self.pose_encoder(x) # [B, L, 64]
        y = self.audio_encoder(y) # [B, L, 64]
        out = torch.cat((x, y), dim=2) # [B, L, 128]
        out = self.decode(out) # [B, L, 64]
        mu = out[:, :, 0::2]
        logvar = out[:, :, 1::2]

        return mu, logvar # [B, L, 32]


if __name__ == "__main__":
    model = Global_VAE(27, 29)
    x = torch.randn(200, 30, 12, 27)
    y = torch.randn(200, 30, 13, 29)
    mu, logvar = model(x, y)
    print(mu.shape)