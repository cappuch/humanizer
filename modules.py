import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.bn1 = nn.GroupNorm(8, out_channels)
        self.bn2 = nn.GroupNorm(8, out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        # x: [B, C, L]
        # t: [B, TimeDim]
        h = self.conv1(F.silu(x))
        h = self.bn1(h)
        
        t_emb = self.mlp(t).unsqueeze(-1)
        h = h + t_emb
        
        h = self.conv2(F.silu(h))
        h = self.bn2(h)
        
        return h + self.shortcut(x)

class UNet1D(nn.Module):
    def __init__(self, seq_len=64, in_channels=2, cond_channels=2, base_channels=32):
        super().__init__()
        self.seq_len = seq_len
        
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        self.cond_scale = nn.Parameter(torch.ones(1))
        
        self.init_conv = nn.Conv1d(in_channels + cond_channels, base_channels, 3, padding=1)
        
        self.down1 = ResidualBlock1D(base_channels, base_channels, time_dim)
        self.down2 = ResidualBlock1D(base_channels, base_channels*2, time_dim)
        self.down3 = ResidualBlock1D(base_channels*2, base_channels*4, time_dim)
        
        self.to_down2 = nn.Conv1d(base_channels, base_channels, 4, 2, 1) # 64 -> 32
        self.to_down3 = nn.Conv1d(base_channels*2, base_channels*2, 4, 2, 1) # 32 -> 16
        
        self.mid1 = ResidualBlock1D(base_channels*4, base_channels*4, time_dim)
        self.mid2 = ResidualBlock1D(base_channels*4, base_channels*4, time_dim)
        
        self.up3 = nn.ConvTranspose1d(base_channels*4, base_channels*2, 4, 2, 1) # 16 -> 32
        self.up_block3 = ResidualBlock1D(base_channels*4, base_channels*2, time_dim) # concat skip
        
        self.up2 = nn.ConvTranspose1d(base_channels*2, base_channels, 4, 2, 1) # 32 -> 64
        self.up_block2 = ResidualBlock1D(base_channels*2, base_channels, time_dim) # concat skip
        
        self.up_block1 = ResidualBlock1D(base_channels, base_channels, time_dim)
        
        self.final_conv = nn.Conv1d(base_channels, in_channels, 1)

    def forward(self, x, t, cond):
        # x: [B, 2, L]
        # t: [B]
        # cond: [B, 2]
        
        cond_tiled = cond.unsqueeze(-1).repeat(1, 1, x.shape[-1]) # [B, 2, L]
        x_in = torch.cat([x, cond_tiled], dim=1) # [B, 4, L]
        
        t_emb = self.time_mlp(t) + self.cond_scale * self.cond_mlp(cond)
        
        x1 = self.init_conv(x_in) # [B, 32, 64]
        x1 = self.down1(x1, t_emb)
        
        x2 = self.to_down2(x1) # [B, 32, 32]
        x2 = self.down2(x2, t_emb) # [B, 64, 32]
        
        x3 = self.to_down3(x2) # [B, 64, 16]
        x3 = self.down3(x3, t_emb) # [B, 128, 16]
        
        m = self.mid1(x3, t_emb)
        m = self.mid2(m, t_emb)
        
        u3 = self.up3(m) # [B, 64, 32]
        u3 = torch.cat([u3, x2], dim=1) # 64+64 = 128
        u3 = self.up_block3(u3, t_emb)
        
        u2 = self.up2(u3) # [B, 32, 64]
        u2 = torch.cat([u2, x1], dim=1) # 32+32 = 64
        u2 = self.up_block2(u2, t_emb)
        
        u1 = self.up_block1(u2, t_emb)
        
        return self.final_conv(u1)
