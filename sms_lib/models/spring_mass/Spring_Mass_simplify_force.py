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


@SIMULATOR.register_module()
class Spring_Mass(nn.Module):
    """
    translated：

    translated(translated):forward(..., frame_id=1, make_rest_on_ground=True)
    translated(x translated 1N,translated 0.3s,translated 1 translated）：
    translated A:translated
    sim.set_external_force(force=[1.0, 0.0, 0.0], duration_seconds=0.3, start_frame=1)
    translated B:translated forward translated
    forward(..., ext_force=[1.0, 0.0, 0.0], ext_duration_seconds=0.3, ext_start=1)
    translated，translated shape translated [N,3] translated Tensor,translated set_external_force(..., per_point=True)。"""
    def __init__(self, cfg, xyz: torch.Tensor, init_velocity=None, load_g=None) -> None:
        super().__init__()
        self.name = type(self).__name__
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.eps = 1e-14
        self.edge = 1e-6
        self.k_neighbors = 256 # cfg.K_NEIGHBORS
        self.k_binding = 16 # cfg.K_BINDING
        self.n_step = 100 # cfg.N_STEP
        freq = cfg.DATA.get('FREQ', -1)
        if freq == -1:
            self.dt = 0.03 # cfg.DATA.DT
        else:
            self.dt = 1 / cfg.DATA.FREQ

        # self.bc = cfg.DATA.BC
        # 
        self.bc = [[[0, 0, -0.2], [0, 0, 1]]] # 
        if self.bc[0][1][1] == 1 or self.bc[0][1][1] == -1:
            self.ground_axis = 1
            self.free_axis = [0, 2] # 
            self.g_f = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device)
            if self.bc[0][1][1] == -1:
                self.inverse_axis = True
            else:
                self.inverse_axis = False
        elif self.bc[0][1][2] == 1 or self.bc[0][1][2] == -1:
            self.ground_axis = 2
            self.free_axis = [0, 1]
            self.g_f = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)
            if self.bc[0][1][2] == -1:
                self.inverse_axis = True
            else:
                self.inverse_axis = False
        else:
            raise ValueError()
        self.ground = self.bc[0][0][self.ground_axis]

        # fitting = True # cfg.get('FITTING', True)
        self.stretch_ratios = 0.05 # cfg.get('STRETCH_RATIOS', 0.0)
        self.ratio_factor = 1 # cfg.get('RATIO_FACTOR', 1.0)
        self.damping = cfg.get('DAMPING', False) # 
        self.fix_mass = True # cfg.get('FIX_MASS', False)
        self.fix_damp = True # cfg.get('FIX_DAMP', False)
        self.optim_fixed_damp = False # cfg.get('OPTIM_FIXED_DAMP', False)
        self.fix_k = True # cfg.get('FIX_K', False)
        self.unlinear_foce = True #cfg.get('UNLINEAR_FORCE', True)
        self.power = 0.5 # cfg.get('POWER', 0.5)
        self.optim_g = False # cfg.get('OPTIM_G', False)
        self.soft_k = False # cfg.get('SOFT_K', True)
        self.spring_bc = False # cfg.get('SPRING_BC', False)
        self.single_k = False # cfg.get('SINGLE_K', False)
        if self.optim_g:
            if load_g is not None:
                self.g = nn.Parameter(torch.tensor(load_g, dtype=torch.float32, device=self.device), requires_grad=True)
            else:
                # self.g = nn.Parameter(torch.tensor(cfg.G, dtype=torch.float32), requires_grad=True)
                # self.g = nn.Parameter(torch.tensor([[0, -9.8, 0]], dtype=torch.float32), requires_grad=True)
                self.g = nn.Parameter(torch.tensor([[0, 0, -9.8]], dtype=torch.float32, device=self.device), requires_grad=True)
        else:
            if load_g is not None:
                self.g = torch.tensor(load_g, dtype=torch.float32, device=self.device)
            else:
                # self.g = torch.tensor(cfg.G, dtype=torch.float32)
                # self.g = torch.tensor([0, -9.8, 0], dtype=torch.float32)
                # self.g = torch.tensor([0, 0, -9.8], dtype=torch.float32)
                self.g = torch.tensor([0, 0, -9.8], dtype=torch.float32, device=self.device)
        
        GLOBAL_K = 200
        GLOBAL_M = 1
        # GLOBAL_DAMP = 0.1
        GLOBAL_DAMP = cfg.DATA.get('GLOBAL_DAMP', 0.1)
        self.initialize(xyz)

        if init_velocity is None:
            self.init_velocity = torch.tensor([[0, 0, 0]], dtype=torch.float32, device=self.device)
            self.stage = 'velocity'
        else:
            self.stage = 'dynamic'
            self.optim_dy_velocity = False # cfg.get('OPTIM_DY_VELOCITY', False)
            self.optim_dy_velocity = False
            self.init_velocity = torch.tensor(init_velocity, dtype=torch.float32, device=self.device)

        if self.fix_k:
            if self.single_k:
                self.global_k = torch.log10(torch.tensor(GLOBAL_K, dtype=torch.float32, device=self.device))
            else:
                self.global_k = torch.log10(torch.tensor(GLOBAL_K, dtype=torch.float32, device=self.device)) * \
                                             torch.ones(self.n_points, dtype=torch.float32, device=self.device)  # 
        else:
            self.global_k = torch.log10(torch.tensor(GLOBAL_K, dtype=torch.float32, device=self.device)) * \
                                         torch.ones_like(self.origin_len, dtype=torch.float32, device=self.device)
        if self.fix_mass:
            self.global_m = torch.log10(torch.tensor(GLOBAL_M, dtype=torch.float32, device=self.device)) * \
                                torch.ones(self.n_points, dtype=torch.float32, device=self.device)
        else:
            self.global_m = torch.log10(torch.tensor(GLOBAL_M, dtype=torch.float32, device=self.device)) * \
                                         torch.ones(self.n_points, dtype=torch.float32, device=self.device)
        if self.damping:
            if self.fix_damp:
                if self.optim_fixed_damp:
                    self.damp = torch.exp(torch.tensor(GLOBAL_DAMP, dtype=torch.float32, device=self.device))
                else:
                    self.damp = torch.log10(torch.tensor(GLOBAL_DAMP, dtype=torch.float32, device=self.device)) * \
                                             torch.ones_like(self.origin_len, dtype=torch.float32, device=self.device)  # 
            else:
                self.damp = torch.log10(torch.tensor(GLOBAL_DAMP, dtype=torch.float32, device=self.device)) * \
                                         torch.ones_like(self.origin_len, dtype=torch.float32, device=self.device)

        self.rebound_k = torch.tensor([-1.0], dtype=torch.float32, device=self.device) # 
        self.fric_k    = torch.tensor([-1.0], dtype=torch.float32, device=self.device) 

        if self.soft_k:
            self.n_fix_spring = 16 # cfg.get('N_FIX_SPRING', 16)
            self.soft_vector = torch.tensor([0.0], dtype=torch.float32, device=self.device)
            # 
            # 
            # 

        if self.spring_bc:
            self.k_bc = torch.log10(torch.tensor(cfg.DATA.GLOBAL_K_BC, dtype=torch.float32, device=self.device))

        # 
        self._ext_force = None          # 
        self._ext_start = 1             # 
        self._ext_frames = 0            # 
        self._ext_per_point = False     # 

        # 
        
    def initialize(self, xyz: torch.Tensor):
        self.device = xyz.device
        self.n_points = xyz.shape[0]
        assert self.n_points == 2048, 'Spring_Mass only supports 2048 points now'
        self.init_xyz = xyz.detach().clone()
        self.init_v = torch.zeros_like(self.init_xyz, dtype=torch.float32)

        self.origin_len, self.knn_index, _ = self.knn(self.init_xyz, self.init_xyz, self.k_neighbors, rm_self=True)

        # 

    def knn(self, x: torch.Tensor, ref: torch.Tensor, k, rm_self=False, sqrt_dist=True):
        """
        :param 
            x: [N, 3]
        :return: 
            dist: [N, k]
            knn_dix: [N, k]
            x_neighbor: [N, k, 3]
        """
        if rm_self:
            # self.k_neighbors + 1: because the first one is the point itself
            dist, knn_dix, x_neighbor = knn_points(x.unsqueeze(0), ref.unsqueeze(0), K=k + 1, return_nn=True)
            dist = dist.squeeze(0)[:, 1:]  # [N, k]
            knn_dix = knn_dix.squeeze(0)[:, 1:]  # [N, k]
            x_neighbor = x_neighbor.squeeze(0)[:, 1:]  # [N, k, 3]
        else:
            dist, knn_dix, x_neighbor = knn_points(x.unsqueeze(0), ref.unsqueeze(0), K=k, return_nn=True)
            dist = dist.squeeze(0)  # [N, k]
            knn_dix = knn_dix.squeeze(0)  # [N, k]
            x_neighbor = x_neighbor.squeeze(0)  # [N, k, 3]

        if sqrt_dist:
            return torch.sqrt(dist), knn_dix, x_neighbor
        else:
            return dist, knn_dix, x_neighbor

    def set_all_particle(self, xyz_all: torch.Tensor):
        self.init_xyz_all = xyz_all.detach().clone()  # 
        self.n_all = self.init_xyz_all.shape[0]

        intrp_len, self.intrp_index, _ = self.knn(self.init_xyz_all, self.init_xyz, self.k_binding)
        intrp_coef = 1 / (intrp_len**0.5 + self.eps)
        self.intrp_coef = intrp_coef / (torch.sum(intrp_coef, dim=-1, keepdim=True))  # + self.eps)
        assert self.intrp_coef.sum().int() == self.init_xyz_all.shape[0], \
            "if report an error here, decrease K_BINDING can help"

    def interpolate(self, xyz_all: torch.Tensor, xyz_before: torch.Tensor, delta_xyz: torch.Tensor):
        delta_knn = delta_xyz[self.intrp_index]  # [N, n, 3] = [238656, 16, 3]
        if self.stage == 'velocity':
            delta_xyz_all = torch.sum(delta_knn * self.intrp_coef.unsqueeze(-1), dim=1)  # [N, 3]
            xyz_all = xyz_all + delta_xyz_all

        elif self.stage == 'dynamic':
            xyz = xyz_before + delta_xyz
            xyz_knn = xyz[self.intrp_index]  # [N, n, 3]
            xyz_all = torch.sum(xyz_knn * self.intrp_coef.unsqueeze(-1), dim=1)  # [N, 3]

        else:
            raise ValueError()

        return xyz_all

    def compute_force(self, xyz, v, K, damp):
        knn_xyz = xyz[self.knn_index]  # [N, k, 3]
        delta_pos = knn_xyz - xyz.unsqueeze(1)  # [N, k, 3]
        curr_len = torch.norm(delta_pos, dim=2)  # [N, k]
        norm_delta_pos = delta_pos / (curr_len.unsqueeze(2) + self.eps)  # [N, k, 3]

        delta_len = (curr_len - self.origin_len)
        delta_len[(delta_len > -self.edge) & (delta_len < self.edge)] = 0.0
        force = (delta_len * K).unsqueeze(2) * norm_delta_pos  # [N, k, 3]

        if self.unlinear_foce and self.power > 0:
            force = force * (1 + torch.abs(curr_len / (self.origin_len + self.eps) - 1).unsqueeze(2))**self.power

        if self.ratio_factor > 1:
            judge = curr_len / (self.origin_len + self.eps) - 1
            mask = torch.where((judge < -self.stretch_ratios) | (judge > self.stretch_ratios))
            force[mask] = self.ratio_factor * force[mask]

        if self.damping:
            knn_v = v[self.knn_index]  # [N, k, 3]
            delta_v = knn_v - v.unsqueeze(1)  # [N, k, 3]
            damp_force = (damp *
                          torch.sum(delta_v * norm_delta_pos, dim=-1)).unsqueeze(-1) * norm_delta_pos  # [N, k, 3]
            force = force + damp_force

        return force.sum(dim=1)

    def apply_bc(self, xyz, v, rebound_k, fric_k):
        # 
        if self.inverse_axis:
            v_index = torch.where(xyz[:, self.ground_axis] >= self.ground)[0]  # 
        else:
            v_index = torch.where(xyz[:, self.ground_axis] <= self.ground)[0]  # 
        # 
        for axis in self.free_axis:
            v[v_index, axis] = fric_k * v[v_index, axis]  # 

        v[v_index, self.ground_axis] = torch.zeros_like(v_index, device=self.device, dtype=torch.float32) # 
        xyz[v_index, self.ground_axis] = torch.zeros_like(v_index, device=self.device, dtype=torch.float32) + self.ground # 

        return xyz, v

    def apply_bc_force(self, force, xyz, v, fric_k):
        k_bc = 10**self.k_bc / (self.origin_len.mean() + self.eps)

        if self.inverse_axis:
            f_index = torch.where(xyz[:, self.ground_axis] >= self.ground)[0]
        else:
            f_index = torch.where(xyz[:, self.ground_axis] <= self.ground)[0]

        force_f = -k_bc * (xyz[f_index, self.ground_axis] - self.ground)
        if self.unlinear_foce and self.power > 0:
            force_f = force_f * torch.abs(xyz[f_index, self.ground_axis] - self.ground)**self.power

        force[f_index, self.ground_axis] = force[f_index, self.ground_axis] + force_f
        for axis in self.free_axis:
            frik_force = False
            if frik_force:
                v_direct = v[f_index, axis]
                v_direct = torch.where(
                    v_direct > 0,
                    torch.tensor(1.0).to(self.device),
                    torch.where(v_direct < 0,
                                torch.tensor(-1.0).to(self.device),
                                torch.tensor(0.0).to(self.device)))
                force[f_index, axis] = force[f_index, axis] - force[f_index, axis] * fric_k * v_direct
            else:
                v[f_index, axis] = fric_k * v[f_index, axis]

        return force, xyz, v

    def step(self, xyz, v, K, m, rebound_k, fric_k, damp, dt, ext_force=None):
        """
        :param
            dt: float
            g: [3]
            xyz: [N, 3]
            knn_xyz: [N, k, 3]
        :medium:
            force: [N, k+1, 3], 1 is gravity
            force_sum: [N, 3]
        :return:
            xyz: [N, 3]
            v: [N, 3]
        """

        # compute_force
        force = self.compute_force(xyz=xyz, v=v, K=K, damp=damp)
        force_sum = force + m.unsqueeze(1) * self.g.unsqueeze(0).to(self.device) * self.g_f.unsqueeze(0).to(self.device)  # [N, 3]

        # 
        if ext_force is not None:
            if ext_force.dim() == 1:
                ext_force = ext_force.unsqueeze(0).repeat(xyz.shape[0], 1)
            force_sum = force_sum + ext_force.to(self.device)

        if self.spring_bc:
            force_sum, xyz, v = self.apply_bc_force(force_sum, xyz, v, fric_k)

        # semi-implicit Euler
        v = v + force_sum * dt / m.unsqueeze(1)  # update velocity
        xyz = xyz + v * dt  # update position

        if not self.spring_bc:
            # Boundary condition
            xyz, v = self.apply_bc(xyz, v, rebound_k, fric_k)

        return xyz, v

    def set_dt(self, freq=None, dt=None):
        if (freq is not None and dt is not None) or (freq is None and dt is None):
            assert False
        if freq is not None:
            self.dt = 1 / freq
        if dt is not None:
            self.dt = dt

    # 
    @torch.no_grad()
    def place_on_ground(self, xyz: torch.Tensor):
        """
        translated xyz，translated（translated，translated ground translated）translated self.ground。
        translated，translated。
        """
        axis = self.ground_axis
        if self.inverse_axis:
            edge_val = torch.max(xyz[:, axis])
        else:
            edge_val = torch.min(xyz[:, axis])
        shift = (self.ground - edge_val).to(xyz.device)
        if torch.abs(shift) > 0:
            xyz = xyz.clone()
            xyz[:, axis] = xyz[:, axis] + shift
        return xyz, shift

    # 
    def set_external_force(self, force, duration_frames=None, duration_seconds=None, start_frame=1, per_point=False):
        """
        force: translated3translated Tensor([3])；translated [N,3]（per_point=True）
        duration_frames: translated（translated forward translated）
        duration_seconds: translated（translated self.dt translated）
        start_frame: translated（>=1）
        per_point: translated
        """
        if isinstance(force, torch.Tensor):
            F = force.to(self.device).float()
        else:
            F = torch.tensor(force, dtype=torch.float32, device=self.device)

        if duration_frames is None and duration_seconds is not None:
            duration_frames = int(np.ceil(duration_seconds / float(self.dt)))
        if duration_frames is None:
            duration_frames = 0

        self._ext_force = F
        self._ext_frames = int(max(0, duration_frames))
        self._ext_start = int(max(1, start_frame))
        self._ext_per_point = bool(per_point)
        return self
    
    def forward(self, xyz_all: torch.Tensor, xyz: torch.Tensor, v: torch.Tensor, frame_id: int, viz=False, **kwargs):
        assert frame_id > 0

        if self.fix_k:
            if self.single_k:
                K = 10**self.global_k.reshape(1, 1).repeat(self.n_points, self.k_neighbors)
            else:
                K = 10**self.global_k.unsqueeze(1).repeat(1, self.k_neighbors)
        else:
            K = 10**self.global_k

        m = 10**self.global_m

        if self.damping:
            if self.optim_fixed_damp:
                damp = torch.log(self.damp) * torch.ones_like(self.origin_len, dtype=torch.float32).to(self.device)
            else:
                damp = 10**self.damp

        rebound_k = torch.sigmoid(self.rebound_k)
        fric_k = torch.clamp(torch.sigmoid(self.fric_k) * 1.2 - 0.1, min=0, max=1)

        K = K / (self.origin_len + self.eps)

        if self.soft_k:
            if self.n_fix_spring == 0:
                soft_learn = torch.arange(self.k_neighbors).to(self.device) / self.k_neighbors
                soft_learn = 2 - torch.pow(torch.exp(torch.nn.functional.softplus(self.soft_vector)), soft_learn)
                k_vector = torch.clamp(soft_learn, 0, 1)
            elif self.n_fix_spring > 0 and self.n_fix_spring < self.k_neighbors:
                soft_learn = torch.arange(
                    (self.k_neighbors - self.n_fix_spring)).to(self.device) / (self.k_neighbors - self.n_fix_spring)
                soft_learn = 2 - torch.pow(torch.exp(torch.nn.functional.softplus(self.soft_vector)), soft_learn)
                soft_learn = torch.clamp(soft_learn, 0, 1)
                k_vector = torch.cat([torch.ones(self.n_fix_spring).to(self.device), soft_learn], dim=0)
            else:
                raise ValueError(f"get invalid n_fix_spring: {self.n_fix_spring}")

            K = K * k_vector.unsqueeze(0)

        if self.damping:
            damp = damp / (self.origin_len + self.eps)
        else:
            damp = None

        # 
        if frame_id == 1 and kwargs.get('make_rest_on_ground', True):
            xyz, shift = self.place_on_ground(xyz)
            if xyz_all is not None and isinstance(xyz_all, torch.Tensor):
                # 
                xyz_all = xyz_all.clone()
                xyz_all[:, self.ground_axis] = xyz_all[:, self.ground_axis] + shift
            # 
            v = torch.zeros_like(v, dtype=torch.float32, device=self.device)

        # 
        xyz_before = xyz.clone()  # 
        
        v = v + self.init_velocity.unsqueeze(0).to(self.device)

        dt = self.dt / self.n_step

        if viz:
            self.viz_step(xyz, v, K, damp, frame_id, **kwargs)

        # 
        ext_force = None
        # 
        kw_ext_force = kwargs.get('ext_force', None)
        kw_ext_frames = kwargs.get('ext_duration_frames', None)
        if kw_ext_frames is None:
            kw_ext_seconds = kwargs.get('ext_duration_seconds', None)
            if kw_ext_seconds is not None:
                kw_ext_frames = int(np.ceil(float(kw_ext_seconds) / float(self.dt)))
        kw_ext_start = kwargs.get('ext_start', None)

        active_force = None
        active_frames = None
        active_start = None

        if kw_ext_force is not None:
            # 
            if isinstance(kw_ext_force, torch.Tensor):
                active_force = kw_ext_force.to(self.device).float()
            else:
                active_force = torch.tensor(kw_ext_force, dtype=torch.float32, device=self.device)
            active_frames = int(max(0, kw_ext_frames or 0))
            active_start = int(max(1, kw_ext_start or 1))
        elif self._ext_force is not None and self._ext_frames > 0:
            # 
            active_force = self._ext_force
            active_frames = self._ext_frames
            active_start = self._ext_start

        # 
        if active_force is not None and active_frames > 0:
            in_window = (frame_id >= active_start) and (frame_id < active_start + active_frames)
            if in_window:
                # 
                if active_force.dim() == 1:
                    ext_force = active_force.unsqueeze(0).repeat(self.n_points, 1)
                else:
                    if active_force.shape[0] == self.n_points:
                        ext_force = active_force
                    else:
                        # 
                        ext_force = active_force.reshape(1, 3).repeat(self.n_points, 1)

        for _ in range(self.n_step):
            xyz, v = self.step(
                xyz=xyz, v=v, K=K, m=m, rebound_k=rebound_k, fric_k=fric_k, damp=damp, dt=dt, ext_force=ext_force
            )
            torch.cuda.empty_cache()

        xyz_all = self.interpolate(xyz_all, xyz_before, delta_xyz=xyz - xyz_before)

        v = v - self.init_velocity.unsqueeze(0).to(self.device)

        is_nan = torch.any(torch.isnan(xyz))
        return xyz_all, xyz, v, is_nan
    
    def set_physics_params(self, params=None, **kwargs):
        if params is None:
            params = kwargs
        else:
            params.update(kwargs)
        device = self.device

        # 
        # 
        if 'k' in params:
            k_value = torch.as_tensor(params['k'], dtype=torch.float32, device=device)
            if self.fix_k:
                if self.single_k:
                    self.global_k = torch.log10(k_value.reshape(1)).squeeze(0)
                else:
                    if k_value.ndim == 0:
                        self.global_k = torch.log10(k_value).repeat(self.n_points)
                    elif k_value.numel() == self.n_points:
                        self.global_k = torch.log10(k_value.reshape(self.n_points))
                    else:
                        raise ValueError("k shape mismatch for fix_k=False,single_k=False")
            else:
                if k_value.ndim == 0:
                    self.global_k = torch.log10(k_value).repeat_as(self.origin_len)
                elif k_value.shape == self.origin_len.shape:
                    self.global_k = torch.log10(k_value)
                else:
                    raise ValueError("k shape must be scalar or match origin_len when fix_k=False")

        # 
        if 'fric_k' in params:
            self.fric_k = torch.as_tensor(params['fric_k'], dtype=torch.float32, device=device).reshape(1)
            
        if 'soft_vector' in params and self.soft_k:
            self.soft_vector = torch.as_tensor([params['soft_vector']], dtype=torch.float32, device=device)
        
        # 
        if 'damp' in params and self.damping:
            damp_value = torch.as_tensor(params['damp'], dtype=torch.float32, device=device)
            if self.fix_damp and self.optim_fixed_damp:
                self.damp = torch.exp(damp_value)
            else:
                if damp_value.ndim == 0:
                    self.damp = torch.log10(damp_value)*torch.ones_like(self.origin_len, dtype=torch.float32, device=self.device)
                elif damp_value.shape == self.origin_len.shape:
                    self.damp = torch.log10(damp_value)
                else:
                    raise ValueError("damp shape must be scalar or match origin_len")

        # 
        # if 'rebound_k' in params:
        #     r = torch.as_tensor(params['rebound_k'], dtype=torch.float32, device=device)
        #     self.rebound_k = r.reshape(1)
        
        # 
        if 'm' in params:
            m_value = torch.as_tensor(params['m'], dtype=torch.float32, device=device)
            if m_value.ndim == 0:
                self.global_m = torch.log10(m_value).repeat(self.n_points)
            elif m_value.numel() == self.n_points:
                self.global_m = torch.log10(m_value.reshape(self.n_points))
            else:
                raise ValueError("m shape mismatch")
        
        return self