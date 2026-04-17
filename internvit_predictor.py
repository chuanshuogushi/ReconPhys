from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

class Bottleneck3D(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=(1,1,1), norm_layer=nn.BatchNorm3d):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes)

        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != (1,1,1) or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * self.expansion),
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


class VideoPhysicsPredictor_internvit300m(nn.Module):
    """
    translated InternViT-300M translated patch-token translated -> translated [B, C, T, H_p, W_p]
    -> 3D ResNet translated -> translated -> MLP translated
    """
    def __init__(self,
                 input_channels=3,
                 hidden_dim=1024,          # 
                 n_points=2048,
                 internvit_model_name="OpenGVLab/InternViT-300M-448px-V2_5",
                 freeze_backbone=True,     # 
                 base_planes=64,           # 
                 depths=(2, 2, 3, 2),      # 
                 dropout_p=0.1,
                 ):
        super().__init__()
        assert input_channels == 3, "InternViT translated 3"
        self.n_points = n_points
        self.dropout_p = dropout_p
        # 
        local_files_only = bool(int(os.environ.get("RECONPHYS_LOCAL_FILES_ONLY", "0")))
        self.backbone = AutoModel.from_pretrained(
            internvit_model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
            torch_dtype=torch.bfloat16,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 
        cfg = getattr(self.backbone, "config", None)
        self.bb_hidden = None
        for k in ["hidden_size", "embed_dim", "width"]:
            if cfg is not None and hasattr(cfg, k):
                self.bb_hidden = getattr(cfg, k)
                break
        if self.bb_hidden is None:
            self.bb_hidden = 1024  # 
        self.bb_image_size = getattr(cfg, "image_size", 448)

        # 
        self.inplanes = base_planes
        self.reduce = nn.Conv3d(self.bb_hidden, self.inplanes, kernel_size=1, bias=False)
        self.reduce_bn = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # 
        norm_layer = nn.BatchNorm3d
        self.layer1 = self._make_layer(Bottleneck3D, planes=base_planes,   blocks=depths[0], stride=(1,1,1), norm_layer=norm_layer)
        self.layer2 = self._make_layer(Bottleneck3D, planes=base_planes*2, blocks=depths[1], stride=(1,2,2), norm_layer=norm_layer)
        self.layer3 = self._make_layer(Bottleneck3D, planes=base_planes*4, blocks=depths[2], stride=(1,2,2), norm_layer=norm_layer)
        self.layer4 = self._make_layer(Bottleneck3D, planes=base_planes*4, blocks=depths[3], stride=(1,2,2), norm_layer=norm_layer)

        self.out_channels = (base_planes*4) * Bottleneck3D.expansion
        self.dropout = nn.Dropout(p=self.dropout_p)

        # 
        feature_dim = self.out_channels
        self.scalar_params_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 128)  # 
        )

        # 
        self.k_head = nn.Linear(128, 1)
        self.fric_k_head = nn.Linear(128, 1)
        self.damp_head = nn.Linear(128, 1)
        self.m_head = nn.Linear(128, 1)

        # 
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        self._init_weights()

        # 
        self.register_buffer("bb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1), persistent=False)
        self.register_buffer("bb_std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1), persistent=False)

    def _make_layer(self, block, planes, blocks, stride, norm_layer):
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=(1,1,1), norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _frames_to_internvit_tokens(self, frames_4d):
        """
        frames_4d: [B*T, 3, H, W] -> translated InternViT translated patch tokens（translated CLS）
        returns:
            feat_map: [B*T, C, H_p, W_p]
        """
        # 
        x = F.interpolate(frames_4d, size=(self.bb_image_size, self.bb_image_size), mode="bilinear", align_corners=False) # 
        x = (x - self.bb_mean) / self.bb_std  
        x = x.to(dtype=torch.bfloat16)
        outputs = self.backbone(pixel_values=x)
        hidden = outputs.last_hidden_state           # [B*T, 1+N, C]
        tokens = hidden[:, 1:, :]                    # 
        c = tokens.size(-1)
        n = tokens.size(1)
        h_p = w_p = int(n ** 0.5)
        tokens = tokens.transpose(1, 2).contiguous() # [B*T, C, N]
        feat_map = tokens.view(tokens.size(0), c, h_p, w_p)
        return feat_map, (h_p, w_p)

    def forward(self, video):
        """
        Args:
            video: [B, C(=3), T, H, W]
        Returns:
            physics_params: dict
        """
        B, C, T, H, W = video.shape
        frames = video.permute(0, 2, 1, 3, 4).contiguous().view(B*T, C, H, W)  # [B*T, 3, H, W]
        feat_map_bt, (Hp, Wp) = self._frames_to_internvit_tokens(frames)       # [B*T, Cbb, Hp, Wp]

        # 
        x = feat_map_bt.view(B, T, self.bb_hidden, Hp, Wp).permute(0, 2, 1, 3, 4).contiguous()  # [B, Cbb, T, Hp, Wp]
        x = x.to(torch.float32)  # 
        # 
        x = self.relu(self.reduce_bn(self.reduce(x)))  # [B, base_planes, T, Hp, Wp]

        # 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dropout(x)

        # 
        global_feat = x.mean(dim=[2, 3, 4])  # [B, out_channels]

        # 
        scalar_features = self.scalar_params_mlp(global_feat)

        physics_params = {
            'k':      1000.0 * self.sigmoid(self.k_head(scalar_features)) + 100.0,
            'fric_k': 2.4 * torch.tanh(self.fric_k_head(scalar_features)),
            'damp':   0.1 + 4.9 * self.sigmoid(self.damp_head(scalar_features)),
            'm':      0.2 + 5.8 * self.sigmoid(self.m_head(scalar_features)),
        }

        for key in physics_params:
            physics_params[key] = physics_params[key].squeeze()
        return physics_params


class VideoPhysicsPredictor_internvit300m_temporalattn(VideoPhysicsPredictor_internvit300m):
    """
    translated VideoPhysicsPredictor_internvit300m translated：
    InternViT -> 3D ResNet -> translated Self-Attention -> MLP translated。
    """
    def __init__(self,
                 input_channels=3,
                 hidden_dim=1024,
                 n_points=2048,
                 internvit_model_name="OpenGVLab/InternViT-300M-448px-V2_5",
                 freeze_backbone=True,
                 base_planes=64,
                 depths=(2, 2, 3, 2),
                 dropout_p=0.1,
                 temporal_num_heads=8,
                 temporal_mlp_ratio=2.0):
        super().__init__(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            n_points=n_points,
            internvit_model_name=internvit_model_name,
            freeze_backbone=freeze_backbone,
            base_planes=base_planes,
            depths=depths,
            dropout_p=dropout_p,
        )

        if self.out_channels % temporal_num_heads != 0:
            raise ValueError(
                f"out_channels({self.out_channels}) must be divisible by temporal_num_heads({temporal_num_heads})"
            )

        self.temporal_norm1 = nn.LayerNorm(self.out_channels)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=self.out_channels,
            num_heads=temporal_num_heads,
            dropout=dropout_p,
            batch_first=True,
        )
        self.temporal_norm2 = nn.LayerNorm(self.out_channels)

        temporal_hidden = int(self.out_channels * temporal_mlp_ratio)
        self.temporal_ffn = nn.Sequential(
            nn.Linear(self.out_channels, temporal_hidden),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(temporal_hidden, self.out_channels),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, video):
        """
        Args:
            video: [B, C(=3), T, H, W]
        Returns:
            physics_params: dict
        """
        B, C, T, H, W = video.shape
        frames = video.permute(0, 2, 1, 3, 4).contiguous().view(B*T, C, H, W)  # [B*T, 3, H, W]
        feat_map_bt, (Hp, Wp) = self._frames_to_internvit_tokens(frames)       # [B*T, Cbb, Hp, Wp]

        # 
        x = feat_map_bt.view(B, T, self.bb_hidden, Hp, Wp).permute(0, 2, 1, 3, 4).contiguous()  # [B, Cbb, T, Hp, Wp]
        x = x.to(torch.float32)
        x = self.relu(self.reduce_bn(self.reduce(x)))  # [B, base_planes, T, Hp, Wp]

        # 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dropout(x)

        # 
        temporal_tokens = x.mean(dim=[3, 4]).permute(0, 2, 1).contiguous()

        # 
        attn_in = self.temporal_norm1(temporal_tokens)
        attn_out, _ = self.temporal_attn(attn_in, attn_in, attn_in, need_weights=False)
        temporal_tokens = temporal_tokens + attn_out
        temporal_tokens = temporal_tokens + self.temporal_ffn(self.temporal_norm2(temporal_tokens))

        # 
        global_feat = temporal_tokens.mean(dim=1)  # [B, out_channels]

        scalar_features = self.scalar_params_mlp(global_feat)

        physics_params = {
            'k':      1000.0 * self.sigmoid(self.k_head(scalar_features)) + 100.0,
            'fric_k': 2.4 * torch.tanh(self.fric_k_head(scalar_features)),
            'damp':   0.1 + 4.9 * self.sigmoid(self.damp_head(scalar_features)),
            'm':      0.2 + 5.8 * self.sigmoid(self.m_head(scalar_features)),
        }

        for key in physics_params:
            physics_params[key] = physics_params[key].squeeze()
        return physics_params


class VideoPhysicsPredictor_Token2Point(nn.Module):
    """
    translated InternViT tokens -> Transformer translated N translated point queries
    translated k[N], m[N]，translated per-edge translated damp[N, k_neighbors]（translated）
    """
    def __init__(self,
                 n_points=2048,
                 k_neighbors=256,
                 internvit_model_name="OpenGVLab/InternViT-300M-448px-V2_5",
                 freeze_backbone=True,
                 d_model=512,
                 nhead=8,
                 num_decoder_layers=3,
                 mlp_hidden=512,
                 dropout_p=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.n_points = n_points
        self.k_neighbors = k_neighbors
        self.dropout_p = dropout_p

        # 
        local_files_only = bool(int(os.environ.get("RECONPHYS_LOCAL_FILES_ONLY", "0")))
        self.backbone = AutoModel.from_pretrained(
            internvit_model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
            torch_dtype=torch.bfloat16,
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        cfg = getattr(self.backbone, "config", None)
        self.bb_hidden = None
        for k in ["hidden_size", "embed_dim", "width"]:
            if cfg is not None and hasattr(cfg, k):
                self.bb_hidden = getattr(cfg, k)
                break
        if self.bb_hidden is None:
            self.bb_hidden = 1024
        self.bb_image_size = getattr(cfg, "image_size", 448)

        # 
        self.token_proj = nn.Linear(self.bb_hidden, d_model)
        self.query_embed = nn.Parameter(torch.randn(self.n_points, d_model) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=d_model*4,
                                                   dropout=dropout_p, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 
        self.head_k = nn.Sequential(
            nn.Linear(d_model, mlp_hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(mlp_hidden, 1)
        )
        self.head_m = nn.Sequential(
            nn.Linear(d_model, mlp_hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(mlp_hidden, 1)
        )
        self.head_damp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden//2), nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden//2, 1)
        )
        self.head_fric = nn.Sequential(
            nn.Linear(d_model, mlp_hidden//2), nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden//2, 1)
        )

        # 
        self.register_buffer("bb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1), persistent=False)
        self.register_buffer("bb_std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1), persistent=False)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def _frames_to_internvit_tokens(self, frames_4d):
        """
        frames_4d: [B*T, 3, H, W] -> InternViT patch tokens（translated CLS）
        return:
            tokens: [B*T, Np, C], Hp, Wp
        """
        x = F.interpolate(frames_4d, size=(self.bb_image_size, self.bb_image_size),
                          mode="bilinear", align_corners=False)
        x = (x - self.bb_mean) / self.bb_std
        x = x.to(dtype=torch.bfloat16)
        outputs = self.backbone(pixel_values=x)
        hidden = outputs.last_hidden_state         # [B*T, 1+Np, C]
        tokens = hidden[:, 1:, :]                  # [B*T, Np, C]
        Np = tokens.size(1)
        Hp = Wp = int(Np ** 0.5)
        return tokens, Hp, Wp

    @staticmethod
    def _pos_enc_2d(h, w, d_model, device):
        """
        translated（2D），translated [h*w, d_model]
        """
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        y = y.reshape(-1).float()
        x = x.reshape(-1).float()

        dim = d_model // 2
        div = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
        pe_x = torch.zeros(x.numel(), dim, device=device)
        pe_y = torch.zeros(y.numel(), dim, device=device)

        pe_x[:, 0::2] = torch.sin(x[:, None] * div)
        pe_x[:, 1::2] = torch.cos(x[:, None] * div)
        pe_y[:, 0::2] = torch.sin(y[:, None] * div)
        pe_y[:, 1::2] = torch.cos(y[:, None] * div)

        pe = torch.cat([pe_x, pe_y], dim=-1)  # [h*w, d_model]
        if pe.size(-1) < d_model:
            pad = d_model - pe.size(-1)
            pe = F.pad(pe, (0, pad), mode='constant', value=0)
        return pe  # [h*w, d_model]

    @staticmethod
    def _pos_enc_1d_t(T, d_model, device):
        """
        translated（1D），translated [T, d_model]
        """
        t = torch.arange(T, device=device).float()
        div = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(T, d_model, device=device)
        pe[:, 0::2] = torch.sin(t[:, None] * div)
        pe[:, 1::2] = torch.cos(t[:, None] * div)
        return pe

    def forward(self, video):
        """
        Args:
            video: [B, 3, T, H, W] in [0,1]
        Returns:
            dict:
              - k:      [B, N]
              - m:      [B, N]
              - damp:   [B, N, k_neighbors]  （translated）
              - fric_k_final: [B]            （0~1）
        """
        B, C, T, H, W = video.shape
        frames = video.permute(0, 2, 1, 3, 4).contiguous().view(B*T, C, H, W)  # [B*T, 3, H, W]

        tokens_bt, Hp, Wp = self._frames_to_internvit_tokens(frames)           # [B*T, Np, Cbb]
        Np = Hp * Wp
        tokens_bt = tokens_bt.to(torch.float32)

        # 
        tokens = tokens_bt.view(B, T, Np, -1)
        pos_hw = self._pos_enc_2d(Hp, Wp, self.token_proj.out_features, tokens.device)  # [Np, d_model]
        pos_t  = self._pos_enc_1d_t(T, self.token_proj.out_features, tokens.device)     # [T, d_model]

        pos_hw = pos_hw.unsqueeze(0).unsqueeze(0).expand(B, T, Np, -1)  # [B,T,Np,d_model]
        pos_t  = pos_t.unsqueeze(0).unsqueeze(2).expand(B, T, Np, -1)   # [B,T,Np,d_model]

        tokens = self.token_proj(tokens) + pos_hw + pos_t               # [B,T,Np,d_model]
        memory = tokens.reshape(B, T*Np, -1)                            # [B, T*Np, d_model]

        # queries: [B,N, d_model]
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        # TransformerDecoder
        point_feat = self.decoder(tgt=queries, memory=memory)           # [B, N, d_model]

        # 
        k_raw   = self.head_k(point_feat).squeeze(-1)                   # [B,N]
        m_raw   = self.head_m(point_feat).squeeze(-1)                   # [B,N]
        damp_raw= self.head_damp(point_feat).squeeze(-1)                # [B,N]
        fric_raw= self.head_fric(point_feat).squeeze(-1).mean(dim=1)    # 

        k      = 100.0 + 1000.0 * self.sigmoid(k_raw)                   # [100,1100]
        m      = 0.2   + 5.8   * self.sigmoid(m_raw)                    # [0.2,6.0]
        damp_p = 0.1   + 4.9   * self.sigmoid(damp_raw)                 # [0.1,5.0]
        fric_k_final = self.sigmoid(fric_raw)                           # [0,1]

        # 
        damp = damp_p.unsqueeze(-1).expand(B, self.n_points, self.k_neighbors)

        physics_params = {
            'k': k,                       # [B,N]
            'm': m,                       # [B,N]
            'damp': damp,                 # [B,N,k_neighbors]
            'fric_k_final': fric_k_final  # [B]
        }
        return physics_params
