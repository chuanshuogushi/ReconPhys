# 
import os
import torch
import torch.nn as nn
import numpy as np

from termcolor import colored
from copy import deepcopy
from pytorch3d.ops import knn_points
from sms_lib.utils.builder import SIMULATOR
from sms_lib.utils.logger import logger
from sms_lib.utils.misc import param_size
from sms_lib.utils.net_utils import init_weights

# 
@SIMULATOR.register_module()
class Spring_Mass(nn.Module):

    def __init__(self, cfg, xyz: torch.Tensor, init_velocity=None, load_g=None) -> None:
        super().__init__()
        self.name = type(self).__name__
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.eps = 1e-14
        self.edge = 1e-6
        self.k_neighbors = 256
        self.k_binding = 16
        self.n_step = 100
        freq = cfg.DATA.get('FREQ', -1)
        if freq == -1:
            self.dt = 0.03
        else:
            self.dt = 1 / cfg.DATA.FREQ

        # 
        self.bc = [[[0, 0, -0.2], [0, 0, 1]]]
        if self.bc[0][1][1] == 1 or self.bc[0][1][1] == -1:
            self.ground_axis = 1
            self.free_axis = [0, 2]
            self.g_f = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device)
            self.inverse_axis = (self.bc[0][1][1] == -1)
        elif self.bc[0][1][2] == 1 or self.bc[0][1][2] == -1:
            self.ground_axis = 2
            self.free_axis = [0, 1]
            self.g_f = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)
            self.inverse_axis = (self.bc[0][1][2] == -1)
        else:
            raise ValueError()
        self.ground = self.bc[0][0][self.ground_axis]

        self.stretch_ratios = 0.05
        self.ratio_factor = 1
        self.damping = cfg.get('DAMPING', False)
        self.fix_mass = True
        self.fix_damp = True
        self.optim_fixed_damp = False
        self.fix_k = True
        self.unlinear_foce = True
        self.power = 0.5
        self.optim_g = False
        self.soft_k = False
        self.spring_bc = False
        self.single_k = False

        if self.optim_g:
            if load_g is not None:
                self.g = nn.Parameter(torch.tensor(load_g, dtype=torch.float32, device=self.device), requires_grad=True)
            else:
                self.g = nn.Parameter(torch.tensor([[0, 0, -9.8]], dtype=torch.float32, device=self.device), requires_grad=True)
        else:
            if load_g is not None:
                self.g = torch.tensor(load_g, dtype=torch.float32, device=self.device)
            else:
                self.g = torch.tensor([0, 0, -9.8], dtype=torch.float32, device=self.device)

        GLOBAL_K = 200
        GLOBAL_M = 1
        GLOBAL_DAMP = cfg.DATA.get('GLOBAL_DAMP', 0.1)

        # 
        self.initialize(xyz)

        if init_velocity is None:
            # 
            self.init_velocity = torch.tensor([[0, 0, 0]], dtype=torch.float32, device=self.device)
            self.stage = 'velocity'
        else:
            self.stage = 'dynamic'
            self.optim_dy_velocity = False
            self.init_velocity = torch.tensor(init_velocity, dtype=torch.float32, device=self.device)

        B, N = self.batch_size, self.n_points

        # 
        if self.fix_k:
            if self.single_k:
                self.global_k = torch.log10(torch.tensor(GLOBAL_K, dtype=torch.float32, device=self.device)).view(1).repeat(B, 1)
            else:
                self.global_k = torch.log10(torch.tensor(GLOBAL_K, dtype=torch.float32, device=self.device)).view(1, 1).repeat(B, N)
        else:
            # 
            self.global_k = torch.log10(torch.tensor(GLOBAL_K, dtype=torch.float32, device=self.device)).view(1, 1, 1).repeat(B, N, self.k_neighbors)

        if self.fix_mass:
            self.global_m = torch.log10(torch.tensor(GLOBAL_M, dtype=torch.float32, device=self.device)).view(1, 1).repeat(B, N)
        else:
            self.global_m = torch.log10(torch.tensor(GLOBAL_M, dtype=torch.float32, device=self.device)).view(1, 1).repeat(B, N)

        if self.damping:
            if self.fix_damp:
                if self.optim_fixed_damp:
                    self.damp = torch.exp(torch.tensor(GLOBAL_DAMP, dtype=torch.float32, device=self.device))
                else:
                    # 
                    self.damp = torch.log10(torch.tensor(GLOBAL_DAMP, dtype=torch.float32, device=self.device)).view(1, 1, 1).repeat(B, N, self.k_neighbors)
            else:
                self.damp = torch.log10(torch.tensor(GLOBAL_DAMP, dtype=torch.float32, device=self.device)).view(1, 1, 1).repeat(B, N, self.k_neighbors)

        self.rebound_k = torch.tensor([-1.0], dtype=torch.float32, device=self.device)
        self.fric_k    = torch.tensor([-1.0], dtype=torch.float32, device=self.device)

        if self.soft_k:
            self.n_fix_spring = 16
            self.soft_vector = torch.tensor([0.0], dtype=torch.float32, device=self.device)

        if self.spring_bc:
            self.k_bc = torch.log10(torch.tensor(cfg.DATA.GLOBAL_K_BC, dtype=torch.float32, device=self.device))

    def _ensure_batch3(self, x: torch.Tensor):
        """
        translated [N,3] translated [B,N,3] translated [B,N,3]，translated batch translated。
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1,N,3]
            return x, True
        elif x.dim() == 3:
            return x, False
        else:
            raise ValueError(f"Expect shape [N,3] or [B,N,3], got {list(x.shape)}")

    def initialize(self, xyz: torch.Tensor):
        xyz, squeezed = self._ensure_batch3(xyz)
        self.device = xyz.device
        self.batch_size = xyz.shape[0]
        self.n_points = xyz.shape[1]
        assert self.n_points == 2048, 'Spring_Mass only supports 2048 points now'
        self.init_xyz = xyz.detach().clone()               # [B,N,3]
        self.init_v = torch.zeros_like(self.init_xyz, dtype=torch.float32)  # [B,N,3]

        # 
        self.origin_len, self.knn_index, _ = self.knn(self.init_xyz, self.init_xyz, self.k_neighbors, rm_self=True)  # [B,N,k], [B,N,k]
        # 

    def knn(self, x: torch.Tensor, ref: torch.Tensor, k, rm_self=False, sqrt_dist=True):
        """
        x:   [B, Nx, 3]
        ref: [B, Nr, 3]
        return:
            dist: [B, Nx, k]
            knn_idx: [B, Nx, k]
            x_neighbor: [B, Nx, k, 3]
        """
        if rm_self:
            dist, knn_idx, x_neighbor = knn_points(x, ref, K=k + 1, return_nn=True)
            # 
            dist = dist[:, :, 1:]           # [B,Nx,k]
            knn_idx = knn_idx[:, :, 1:]     # [B,Nx,k]
            x_neighbor = x_neighbor[:, :, 1:, :]  # [B,Nx,k,3]
        else:
            dist, knn_idx, x_neighbor = knn_points(x, ref, K=k, return_nn=True)  # [B,Nx,k], [B,Nx,k], [B,Nx,k,3]
        if sqrt_dist:
            return torch.sqrt(dist + 1e-12), knn_idx, x_neighbor
        else:
            return dist, knn_idx, x_neighbor

    # def _batch_gather_knn(self, x: torch.Tensor, knn_idx: torch.Tensor):
    #     """
    #     x: [B,N,3], knn_idx: [B,N,k] -> [B,N,k,3]
    #     """
    #     B, N, _ = x.shape  # [1,2048,3]
    #     _, _, K = knn_idx.shape  # [1,238656,16]
    #     x_expand = x.unsqueeze(1).expand(B, N, N, 3)              # [B,N,N,3]
    #     idx_expand = knn_idx.unsqueeze(-1).expand(B, N, K, 3)     # [B,N,K,3]
    #     gathered = torch.gather(x_expand, 2, idx_expand)          # [B,N,K,3]
    #     return gathered
    def _batch_gather_knn(self, x: torch.Tensor, knn_idx: torch.Tensor):
        """
        translated x translated knn_idx translated1translated（batchtranslated），translated M translated。
        Args:
            x:       [B, N, C]  （translated C=3）
            knn_idx: [B, M, K]  （M translated N translated N_all）
        Returns:
            gathered: [B, M, K, C]
        """
        assert x.dim() == 3 and knn_idx.dim() == 3, f"shape invalid: x={x.shape}, knn_idx={knn_idx.shape}"
        B, N, C = x.shape
        B2, M, K = knn_idx.shape
        assert B == B2, "batch size mismatch"

        idx = knn_idx.long()  # 
        # 
        batch_offset = (torch.arange(B, device=x.device).view(B, 1, 1) * N)  # [B,1,1]
        idx_flat = (idx + batch_offset).reshape(-1)                          # [B*M*K]

        x_flat = x.reshape(B * N, C)                                         # [B*N, C]
        gathered = x_flat[idx_flat].view(B, M, K, C)                         # [B, M, K, C]
        return gathered
    def set_all_particle(self, xyz_all: torch.Tensor):
        """
        xyz_all: [N_all,3] translated [B,N_all,3]
        """
        xyz_all, squeezed = self._ensure_batch3(xyz_all)  # [B,N_all,3]
        self.init_xyz_all = xyz_all.detach().clone()
        self.n_all = self.init_xyz_all.shape[1]

        # 
        intrp_len, self.intrp_index, _ = self.knn(self.init_xyz_all, self.init_xyz, self.k_binding)  # [B,N_all,kb]
        intrp_coef = 1 / (intrp_len**0.5 + self.eps)  # [B,N_all,kb]
        self.intrp_coef = intrp_coef / (torch.sum(intrp_coef, dim=-1, keepdim=True) + self.eps)  # 
        return self

    def interpolate(self, xyz_all: torch.Tensor, xyz_before: torch.Tensor, delta_xyz: torch.Tensor):
        """
        xyz_all:   [B,N_all,3]
        xyz_before:[B,N,3]
        delta_xyz: [B,N,3]
        """
        # 
        delta_knn = self._batch_gather_knn(delta_xyz, self.intrp_index)  # [B,N_all,kb,3]
        if self.stage == 'velocity':
            delta_xyz_all = torch.sum(delta_knn * self.intrp_coef.unsqueeze(-1), dim=2)  # [B,N_all,3]
            xyz_all = xyz_all + delta_xyz_all
        elif self.stage == 'dynamic':
            xyz = xyz_before + delta_xyz
            xyz_knn = self._batch_gather_knn(xyz, self.intrp_index)  # [B,N_all,kb,3]
            xyz_all = torch.sum(xyz_knn * self.intrp_coef.unsqueeze(-1), dim=2)  # [B,N_all,3]
        else:
            raise ValueError()
        return xyz_all

    def compute_force(self, xyz, v, K_scaled, damp_scaled=None):
        """
        xyz:        [B,N,3]
        v:          [B,N,3]
        K_scaled:   [B,N,k]   (translated origin_len)
        damp_scaled:[B,N,k] translated None  (translated origin_len)
        return:
            force_sum_neighbors: [B,N,3]
        """
        knn_xyz = self._batch_gather_knn(xyz, self.knn_index)          # [B,N,k,3]
        delta_pos = knn_xyz - xyz.unsqueeze(2)                         # [B,N,k,3]
        curr_len = torch.norm(delta_pos, dim=3)                        # [B,N,k]
        norm_delta_pos = delta_pos / (curr_len.unsqueeze(-1) + self.eps)  # [B,N,k,3]

        delta_len = (curr_len - self.origin_len)                       # [B,N,k]
        edge_mask = (torch.abs(delta_len) < self.edge)
        delta_len = torch.where(edge_mask, torch.zeros_like(delta_len), delta_len)

        force = (delta_len * K_scaled).unsqueeze(-1) * norm_delta_pos  # [B,N,k,3]

        if self.unlinear_foce and self.power > 0:
            factor = (1 + torch.abs(curr_len / (self.origin_len + self.eps) - 1)) ** self.power  # [B,N,k]
            force = force * factor.unsqueeze(-1)

        if self.ratio_factor > 1:
            judge = curr_len / (self.origin_len + self.eps) - 1        # [B,N,k]
            mask = (judge < -self.stretch_ratios) | (judge > self.stretch_ratios)
            force = torch.where(mask.unsqueeze(-1), self.ratio_factor * force, force)

        if self.damping and damp_scaled is not None:
            knn_v = self._batch_gather_knn(v, self.knn_index)          # [B,N,k,3]
            delta_v = knn_v - v.unsqueeze(2)                            # [B,N,k,3]
            proj = torch.sum(delta_v * norm_delta_pos, dim=-1)          # [B,N,k]
            damp_force = (damp_scaled * proj).unsqueeze(-1) * norm_delta_pos
            force = force + damp_force

        return force.sum(dim=2)  # [B,N,3]

    def apply_bc(self, xyz, v, rebound_k, fric_k):
        """
        translated batch translated。
        """
        if self.inverse_axis:
            mask = (xyz[..., self.ground_axis] >= self.ground)  # [B,N]
        else:
            mask = (xyz[..., self.ground_axis] <= self.ground)

        # 
        for axis in self.free_axis:
            v_axis = v[..., axis]
            v_axis = torch.where(mask, fric_k * v_axis, v_axis)
            v[..., axis] = v_axis

        # 
        v_g = v[..., self.ground_axis]
        v_g = torch.where(mask, torch.zeros_like(v_g), v_g)
        v[..., self.ground_axis] = v_g

        # 
        x_g = xyz[..., self.ground_axis]
        x_g = torch.where(mask, torch.full_like(x_g, self.ground), x_g)
        xyz[..., self.ground_axis] = x_g

        return xyz, v

    def apply_bc_force(self, force, xyz, v, fric_k):
        # 
        k_bc = 10**self.k_bc / (self.origin_len.mean() + self.eps)

        if self.inverse_axis:
            mask = (xyz[..., self.ground_axis] >= self.ground)  # [B,N]
        else:
            mask = (xyz[..., self.ground_axis] <= self.ground)

        # 
        f_norm = -(xyz[..., self.ground_axis] - self.ground)    # [B,N]
        if self.unlinear_foce and self.power > 0:
            f_norm = f_norm * torch.abs(xyz[..., self.ground_axis] - self.ground) ** self.power
        f_norm = k_bc * f_norm                                   # [B,N]

        f_g = force[..., self.ground_axis]
        f_g = torch.where(mask, f_g + f_norm, f_g)
        force[..., self.ground_axis] = f_g

        # 
        for axis in self.free_axis:
            v_axis = v[..., axis]
            v_axis = torch.where(mask, fric_k * v_axis, v_axis)
            v[..., axis] = v_axis

        return force, xyz, v

    @torch.no_grad()
    def viz_step(self, xyz, v, K, damp, frame_id, **kwargs):
        # 
        assert xyz.shape[0] == 1, "viz_step translated batch==1"
        import cv2
        import imageio
        from sms_lib.utils.transform import SE3_transform, persp_project
        from sms_lib.models.gaus.utils.graphics_utils import fov2focal

        viz_force_dir = kwargs['viz_force_dir']
        viewpoint_cam = kwargs['viewpoint_cam']
        os.makedirs(viz_force_dir, exist_ok=True)

        viz_image = kwargs['viz_image']
        viz_image = (viz_image.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
        frame = viz_image.copy()

        R = viewpoint_cam.R.transpose()
        T = viewpoint_cam.T
        w2c = np.concatenate([R, T[:, None]], axis=-1)
        w2c = np.concatenate([w2c, [[0, 0, 0, 1]]], axis=0)

        fx = fov2focal(viewpoint_cam.FoVx, viewpoint_cam.image_width)
        fy = fov2focal(viewpoint_cam.FoVy, viewpoint_cam.image_height)
        intrisic = np.array([
            [fx, 0, viewpoint_cam.image_width / 2],
            [0, fy, viewpoint_cam.image_height / 2],
            [0, 0, 1],
        ])

        xyz0 = xyz[0]
        xyz_c = SE3_transform(xyz0.detach().cpu().numpy(), w2c)
        uv = persp_project(xyz_c, intrisic)

        for i in range(uv.shape[0]):
            cx = int(uv[i, 0])
            cy = int(uv[i, 1])
            cv2.circle(frame, (cx, cy), radius=1, thickness=-1, color=np.array([1.0, 0.0, 0.0]) * 255)

        imageio.imwrite(os.path.join(viz_force_dir, f"{frame_id:02}.png"), frame)

    def step(self, xyz, v, K_scaled, m, rebound_k, fric_k, damp_scaled, dt):
        """
        xyz, v:       [B,N,3]
        K_scaled:     [B,N,k]
        m:            [B,N]
        damp_scaled:  [B,N,k] or None
        """
        force = self.compute_force(xyz=xyz, v=v, K_scaled=K_scaled, damp_scaled=damp_scaled)  # [B,N,3]
        # 
        gravity = m.unsqueeze(-1) * self.g.unsqueeze(0) * self.g_f.unsqueeze(0)  # [B,N,3]
        force_sum = force + gravity

        if self.spring_bc:
            force_sum, xyz, v = self.apply_bc_force(force_sum, xyz, v, fric_k)

        # semi-implicit Euler
        v = v + force_sum * dt / (m.unsqueeze(-1) + self.eps)  # [B,N,3]
        xyz = xyz + v * dt

        if not self.spring_bc:
            xyz, v = self.apply_bc(xyz, v, rebound_k, fric_k)

        return xyz, v

    def set_dt(self, freq=None, dt=None):
        if (freq is not None and dt is not None) or (freq is None and dt is None):
            assert False
        if freq is not None:
            self.dt = 1 / freq
        if dt is not None:
            self.dt = dt

    def forward(self, xyz_all: torch.Tensor, xyz: torch.Tensor, v: torch.Tensor, frame_id: int, viz=False, **kwargs):
        """
        xyz_all: [N_all,3] translated [B,N_all,3]
        xyz, v:  [N,3]     translated [B,N,3]
        translated batch translated。
        """
        assert frame_id > 0

        # 
        xyz_all, squeeze_all = self._ensure_batch3(xyz_all)
        xyz, squeeze_xyz = self._ensure_batch3(xyz)
        v, squeeze_v = self._ensure_batch3(v)
        B = xyz.shape[0]

        # 
        if self.fix_k:
            if self.single_k:
                # [B,1] -> [B,N,k]
                K_param = (10**self.global_k).view(B, 1, 1).repeat(B, self.n_points, self.k_neighbors)
            else:
                # [B,N] -> [B,N,k]
                K_param = (10**self.global_k).unsqueeze(-1).repeat(1, 1, self.k_neighbors)
        else:
            # [B,N,k]
            K_param = (10**self.global_k)

        m = 10**self.global_m  # [B,N]

        if self.damping:
            if self.fix_damp and self.optim_fixed_damp:
                damp_param = torch.log(self.damp) * torch.ones_like(self.origin_len, dtype=torch.float32).to(self.device)
            else:
                damp_param = 10**self.damp  # 
        else:
            damp_param = None

        rebound_k = torch.sigmoid(self.rebound_k)  # 
        fric_k = torch.clamp(torch.sigmoid(self.fric_k) * 1.2 - 0.1, min=0, max=1)  # 

        # 
        K_scaled = K_param / (self.origin_len + self.eps)  # [B,N,k]
        if self.damping and damp_param is not None:
            damp_scaled = damp_param / (self.origin_len + self.eps)  # [B,N,k]
        else:
            damp_scaled = None

        xyz_before = xyz.clone()  # [B,N,3]

        v = v + self.init_velocity.view(1, 1, 3).to(self.device)  # 

        dt = self.dt / self.n_step

        if viz:
            self.viz_step(xyz, v, K_scaled, damp_scaled, frame_id, **kwargs)

        for _ in range(self.n_step):
            xyz, v = self.step(xyz=xyz, v=v, K_scaled=K_scaled, m=m, rebound_k=rebound_k, fric_k=fric_k, damp_scaled=damp_scaled, dt=dt)
            torch.cuda.empty_cache()

        xyz_all = self.interpolate(xyz_all, xyz_before, delta_xyz=xyz - xyz_before)  # [B,N_all,3]

        v = v - self.init_velocity.view(1, 1, 3).to(self.device)

        is_nan = torch.any(torch.isnan(xyz))

        # 
        if squeeze_all:
            xyz_all = xyz_all.squeeze(0)
        if squeeze_xyz:
            xyz = xyz.squeeze(0)
        if squeeze_v:
            v = v.squeeze(0)

        return xyz_all, xyz, v, is_nan

    def set_physics_params(self, params=None, **kwargs):
        """
        translated、[B]、[B,N]、[B,N,k] translated，translated batch translated。
        """
        if params is None:
            params = kwargs
        else:
            params.update(kwargs)
        device = self.device
        B, N = self.batch_size, self.n_points

        def _to_batch_point(x, like='point'):
            # like='point' -> [B,N]；like='edge' -> [B,N,k]
            if isinstance(x, (float, int)):
                x = torch.tensor([x], dtype=torch.float32, device=device)
            x = torch.as_tensor(x, dtype=torch.float32, device=device)
            if like == 'point':
                if x.dim() == 0:
                    x = x.view(1, 1).repeat(B, N)
                elif x.dim() == 1:
                    if x.numel() == 1:
                        x = x.view(1, 1).repeat(B, N)
                    elif x.numel() == B:
                        x = x.view(B, 1).repeat(1, N)
                    else:
                        raise ValueError("shape mismatch for point-like param")
                elif x.dim() == 2:
                    assert x.shape == (B, N), "expect [B,N]"
                else:
                    raise ValueError("invalid dim for point-like")
            else:
                if x.dim() == 0:
                    x = x.view(1, 1, 1).repeat(B, N, self.k_neighbors)
                elif x.dim() == 1:
                    if x.numel() == 1:
                        x = x.view(1, 1, 1).repeat(B, N, self.k_neighbors)
                    elif x.numel() == B:
                        x = x.view(B, 1, 1).repeat(1, N, self.k_neighbors)
                    else:
                        raise ValueError("shape mismatch for edge-like param")
                elif x.dim() == 2:
                    if x.shape == (B, N):
                        x = x.unsqueeze(-1).repeat(1, 1, self.k_neighbors)
                    else:
                        raise ValueError("shape mismatch for edge-like param")
                elif x.dim() == 3:
                    assert x.shape == (B, N, self.k_neighbors), "expect [B,N,k]"
                else:
                    raise ValueError("invalid dim for edge-like")
            return x

        # 
        if 'k' in params:
            k_value = params['k']
            if self.fix_k:
                if self.single_k:
                    k_b = _to_batch_point(k_value, like='point')  # [B,N]
                    # 
                    self.global_k = torch.log10(k_b.mean(dim=1, keepdim=True))  # [B,1]
                else:
                    k_b = _to_batch_point(k_value, like='point')  # [B,N]
                    self.global_k = torch.log10(k_b)  # [B,N]
            else:
                k_e = _to_batch_point(k_value, like='edge')  # [B,N,k]
                self.global_k = torch.log10(k_e)  # [B,N,k]

        # 
        if 'fric_k' in params:
            fk = torch.as_tensor(params['fric_k'], dtype=torch.float32, device=device).reshape(1)
            self.fric_k = fk

        # 
        if 'damp' in params and self.damping:
            damp_value = params['damp']
            if self.fix_damp and self.optim_fixed_damp:
                dv = torch.as_tensor(damp_value, dtype=torch.float32, device=device)
                self.damp = torch.exp(dv)
            else:
                d_e = _to_batch_point(damp_value, like='edge')  # 
                self.damp = torch.log10(d_e)

        # 
        if 'm' in params:
            m_value = _to_batch_point(params['m'], like='point')  # [B,N]
            self.global_m = torch.log10(m_value)

        # soft_vector
        if 'soft_vector' in params and self.soft_k:
            self.soft_vector = torch.as_tensor([params['soft_vector']], dtype=torch.float32, device=device)

        return self