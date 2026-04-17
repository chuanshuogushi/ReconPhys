# Take one mp4 video, predict falling dynamics, and render the simulated result from 4 views.
import os
import argparse
import random
from pathlib import Path
import numpy as np
import torch
import imageio
from PIL import Image
from yacs.config import CfgNode as CN

THIS_DIR = Path(__file__).resolve().parent
from trellis.representations.gaussian.gaussian_model import Gaussian
from trellis.utils import render_utils
from sms_lib.utils.transform import uniform_sampling
from sms_lib.utils.builder import build_simulator

from release_utils import load_video_as_tensor
from internvit_predictor import (
    VideoPhysicsPredictor_internvit300m,
    VideoPhysicsPredictor_internvit300m_temporalattn,
    VideoPhysicsPredictor_Token2Point,
)


def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_simulator_from_prediction(cfg, xyz):
    from sms_lib.models.spring_mass.Spring_Mass_simplify import Spring_Mass  # noqa: F401
    velocity = [0, 0, 0]
    cfg.DAMPING = True
    simulator = build_simulator(
        cfg.DYNAMIC,
        xyz=xyz,
        data=cfg.DATA,
        init_velocity=velocity,
        load_g=None,
    )
    return simulator.cuda()


def load_checkpoint_weights(ckpt_path, model, device):
    state = torch.load(ckpt_path, map_location=device)
    sd = state.get("model", state)
    model.load_state_dict(sd, strict=True)


def _to_cpu_for_save(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu_for_save(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_for_save(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_for_save(v) for v in obj)
    return obj


def infer_gaussian_path(input_mp4: str, gaussian_ply: str = "") -> str:
    if gaussian_ply:
        return gaussian_ply
    obj_dir = os.path.dirname(input_mp4)
    guess = os.path.join(obj_dir, "gaussian.ply")
    return guess


def _sample_frames_from_mp4(input_mp4: str):
    reader = imageio.get_reader(input_mp4)
    try:
        frame0 = reader.get_data(0)
    except Exception as e:
        raise RuntimeError(f"Failed to read first frame from mp4: {input_mp4}") from e
    return Image.fromarray(frame0).convert("RGB")


def generate_gaussian_from_video(input_mp4: str, out_ply: str, trellis_model_dir: str):
    from trellis.pipelines import TrellisImageTo3DPipeline

    os.makedirs(os.path.dirname(out_ply) or ".", exist_ok=True)
    print(f"[Info] gaussian.ply not found, generating from video: {input_mp4}")

    image = _sample_frames_from_mp4(input_mp4)

    pipeline = TrellisImageTo3DPipeline.from_pretrained(trellis_model_dir)
    pipeline.cuda()

    outputs = pipeline.run(
        image,
        seed=1,
        sparse_structure_sampler_params={
            "steps": 12,
            "cfg_strength": 7.5,
        },
        slat_sampler_params={
            "steps": 12,
            "cfg_strength": 3,
        },
    )
    outputs['gaussian'][0].save_ply(out_ply)
    print(f"[OK] generated gaussian: {out_ply}")


def build_predictor(model_name: str, internvit_model_name: str, device: torch.device):
    if model_name == "internvit300m":
        model = VideoPhysicsPredictor_internvit300m(internvit_model_name=internvit_model_name)
    elif model_name == "token2point":
        model = VideoPhysicsPredictor_Token2Point(internvit_model_name=internvit_model_name)
    elif model_name == "internvit300m_temporalattn":
        model = VideoPhysicsPredictor_internvit300m_temporalattn(internvit_model_name=internvit_model_name)
    return model.to(device)


def _resolve_existing_path(path_str: str, candidates: list[Path], expect: str) -> str:
    p = Path(path_str).expanduser()
    if p.exists():
        return str(p)
    if not p.is_absolute():
        for base in candidates:
            cand = (base / p).expanduser()
            if cand.exists():
                return str(cand)
    kind = "file" if expect == "file" else "directory"
    hint = "\n".join([f"- {str(c)}" for c in candidates])
    raise FileNotFoundError(
        f"Cannot find {kind}: {path_str}\n"
        f"Tried these base paths:\n{hint}"
    )


@torch.no_grad()
def run_demo(
    predictor,
    cfg,
    video_tensor,
    gaussian_ply,
    total_frames=30,
    in_frames=20,
    voxel_size=0.01,
    ctrl_points=2048,
    nstep_retry=3,
    step_add=50,
    save_params_path="",
):
    predictor.eval()

    predictor_input = video_tensor[:, :, :in_frames, :, :]
    pred_params = predictor(predictor_input)
    if save_params_path:
        os.makedirs(os.path.dirname(save_params_path) or ".", exist_ok=True)
        torch.save(_to_cpu_for_save(pred_params), save_params_path)
        print(f"[Info] saved predicted physics params: {save_params_path}")
    gaussians = Gaussian(
        sh_degree=0,
        aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
        mininum_kernel_size=0.0009,
        scaling_bias=0.004,
        opacity_bias=0.1,
        scaling_activation='softplus',
    )
    gaussians.load_ply(gaussian_ply)
    xyz_all = gaussians._xyz

    xyz = uniform_sampling(xyz_all, voxel_size=voxel_size)
    n_ctrl = min(ctrl_points, xyz.shape[0])
    idx = random.sample(range(xyz.shape[0]), n_ctrl)
    xyz = xyz[idx, :]

    simulator = setup_simulator_from_prediction(cfg, xyz.squeeze(0))
    simulator.set_all_particle(xyz_all)
    simulator.set_physics_params(pred_params)
    
    v = simulator.init_v.detach().clone()

    # Save 4 independent views per frame
    frames_by_view = [[], [], [], []]

    for frame_id in range(total_frames):
        if frame_id > 0:
            ok = False
            for _ in range(nstep_retry):
                xyz_all_o, xyz_o, v_o, is_nan = simulator(xyz_all, xyz, v, frame_id)
                if not is_nan:
                    ok = True
                    break
                simulator.n_step += step_add

            if not ok:
                print(f"[WARN] NaN persists at frame={frame_id}, stop simulation early.")
                break

            gaussians._xyz = xyz_all_o
            xyz = xyz_o.detach().clone()
            xyz_all = xyz_all_o.detach().clone()
            v = v_o.detach().clone()

        # [4, 3, H, W]
        color_4 = render_utils.render_snapshot_with_grad(
            gaussians, offset=(0, 0), r=2, fov=40
        )["color"]
        # -> [4, H, W, 3]
        color_4 = color_4.permute(0, 2, 3, 1).detach().cpu().numpy()
        color_4 = (np.clip(color_4, 0.0, 1.0) * 255.0).astype(np.uint8)

        for vi in range(4):
            frames_by_view[vi].append(color_4[vi])

    # If stopped early due to NaN, pad to total_frames
    if len(frames_by_view[0]) == 0:
        raise RuntimeError("No frame rendered. Please check input video / gaussian / checkpoint.")
    for vi in range(4):
        while len(frames_by_view[vi]) < total_frames:
            frames_by_view[vi].append(frames_by_view[vi][-1])

    return [v[:total_frames] for v in frames_by_view]


def main():
    parser = argparse.ArgumentParser("Demo: mp4 -> falling video (4 views, 30 frames)")
    parser.add_argument("--input_mp4", type=str, required=True, help="input video mp4")
    parser.add_argument("--ckpt", type=str, required=True, help="predictor checkpoint")
    parser.add_argument("--out_mp4", type=str, default="./demo_4view.mp4", help="output demo video path")
    parser.add_argument("--gaussian_ply", type=str, default="", help="optional gaussian.ply path; auto infer if empty")
    parser.add_argument("--trellis_model_dir", type=str, default="microsoft/TRELLIS-image-large", help="TRELLIS model dir for auto 3D-asset generation")
    parser.add_argument("--model", type=str, default="internvit300m_temporalattn", choices=["internvit300m", "internvit300m_temporalattn", "token2point"])
    parser.add_argument("--internvit_model_name", type=str, default="OpenGVLab/InternViT-300M-448px-V2_5")
    parser.add_argument("--cfg_default", type=str, default="default.yaml")
    parser.add_argument("--cfg_scene", type=str, default="multiscene.yaml")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--in_frames", type=int, default=20)
    parser.add_argument("--resize", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--save_params_path", type=str, default="", help="path to save predicted physics params (.pt)")
    args = parser.parse_args()

    file_candidates = [
        Path.cwd(),
        THIS_DIR,
        THIS_DIR / "config" / "mpm_synthetic",
        THIS_DIR / "demos",
    ]

    args.input_mp4 = _resolve_existing_path(args.input_mp4, file_candidates, expect="file")
    args.ckpt = _resolve_existing_path(args.ckpt, file_candidates, expect="file")
    args.cfg_default = _resolve_existing_path(args.cfg_default, file_candidates, expect="file")
    args.cfg_scene = _resolve_existing_path(args.cfg_scene, file_candidates, expect="file")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gaussian_ply = infer_gaussian_path(args.input_mp4, args.gaussian_ply)
    if not os.path.isfile(gaussian_ply):
        generate_gaussian_from_video(args.input_mp4, gaussian_ply, args.trellis_model_dir)
    if not os.path.isfile(gaussian_ply):
        raise FileNotFoundError(f"failed to generate gaussian.ply: {gaussian_ply}")

    os.makedirs(os.path.dirname(args.out_mp4) or ".", exist_ok=True)
    if not args.save_params_path:
        out_root_for_params, _ = os.path.splitext(args.out_mp4)
        args.save_params_path = f"{out_root_for_params}_pred_params.pt"

    predictor = build_predictor(args.model, args.internvit_model_name, device)
    load_checkpoint_weights(args.ckpt, predictor, device)
    predictor.eval()

    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.cfg_default)
    cfg.merge_from_file(args.cfg_scene)

    video_tensor = load_video_as_tensor(
        args.input_mp4, num_frames=args.frames, resize=tuple(args.resize)
    )
    # Support both [3,T,H,W] and [1,3,T,H,W]
    if video_tensor.ndim == 4:
        video_tensor = video_tensor.unsqueeze(0)
    video_tensor = video_tensor.to(device, non_blocking=True)

    frames_by_view = run_demo(
        predictor,
        cfg,
        video_tensor,
        gaussian_ply=gaussian_ply,
        total_frames=args.frames,
        in_frames=args.in_frames,
        step_add=getattr(cfg.DYNAMIC, "STEP_ADD", 50),
        save_params_path=args.save_params_path,
    )

    out_root, out_ext = os.path.splitext(args.out_mp4)
    out_ext = out_ext if out_ext else ".mp4"
    out_paths = []
    for vi in range(4):
        out_path = f"{out_root}_view{vi}{out_ext}"
        imageio.mimwrite(out_path, frames_by_view[vi], fps=args.fps)
        out_paths.append(out_path)

    print("[OK] saved:")
    for p in out_paths:
        print(f"  - {p}")
    print(f"[Info] input: {args.input_mp4}")
    print(f"[Info] gaussian: {gaussian_ply}")


if __name__ == "__main__":
    main()