from __future__ import annotations

from typing import Dict

import imageio
import torch
import torchvision.transforms as T
from PIL import Image


def compute_param_l1_errors(pred_params: Dict, gt_params: Dict) -> Dict[str, float]:
    errs = {}
    for key in ["k", "fric_k", "m", "damp"]:
        if key in pred_params and key in gt_params:
            pred = pred_params[key]
            gt = gt_params[key]
            if not isinstance(pred, torch.Tensor):
                pred = torch.as_tensor(pred)
            if not isinstance(gt, torch.Tensor):
                gt = torch.as_tensor(gt, dtype=pred.dtype, device=pred.device)
            errs[key] = float(torch.mean(torch.abs(pred - gt)).detach().cpu().item())
    return errs


def load_video_as_tensor(video_path: str, num_frames: int = 16, resize=(128, 128)) -> torch.Tensor:
    reader = imageio.get_reader(video_path)
    frames = []
    transform = T.Compose([T.Resize(resize), T.ToTensor()])

    for i, frame in enumerate(reader):
        if i >= num_frames:
            break
        frames.append(transform(Image.fromarray(frame)))
    reader.close()

    if len(frames) == 0:
        raise RuntimeError(f"No frames loaded from video: {video_path}")

    return torch.stack(frames, dim=1)  # [3,T,H,W]


def setup_simulator_from_prediction(cfg, xyz):
    from sms_lib.utils.builder import build_simulator
    # Match the source test path that registers Spring_Mass_simplify.
    from sms_lib.models.spring_mass.Spring_Mass_simplify import Spring_Mass  # noqa: F401

    cfg.DAMPING = True
    velocity = [0, 0, 0]
    simulator = build_simulator(
        cfg.DYNAMIC,
        xyz=xyz,
        data=cfg.DATA,
        init_velocity=velocity,
        load_g=None,
    )
    return simulator.cuda()
