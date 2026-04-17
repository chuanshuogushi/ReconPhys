from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn as nn

from yacs.config import CfgNode as CN

from internvit_predictor import (
    VideoPhysicsPredictor_Token2Point,
    VideoPhysicsPredictor_internvit300m,
    VideoPhysicsPredictor_internvit300m_temporalattn,
)
from psnr_ssim_lpips import VideoEvaluator, chamfer_distance, earth_movers_distance, subsample_points
from release_utils import compute_param_l1_errors, load_video_as_tensor, setup_simulator_from_prediction


THIS_DIR = Path(__file__).resolve().parent


def load_checkpoint_weights(ckpt_path, model, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        sd = state["model"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        sd = state["model_state_dict"]
    else:
        sd = state
    model.load_state_dict(sd, strict=True)
    return state


@torch.no_grad()
def infer_one_video(video_tensor, predictor, cfg, gs_path_obj, num_frames):
    from trellis.representations.gaussian.gaussian_model import Gaussian
    from trellis.utils import render_utils
    from sms_lib.utils.transform import uniform_sampling

    predictor.eval()

    rec_frame_startend = [0, 20]
    predictor_input = video_tensor[:, :, rec_frame_startend[0] : rec_frame_startend[1], :, :]
    pred_params_raw = predictor(predictor_input)
    pred_params = {}
    for k, v in pred_params_raw.items():
        vv = v
        if isinstance(vv, torch.Tensor) and vv.ndim >= 1 and vv.shape[0] == 1:
            vv = vv.squeeze(0)
        pred_params[k] = vv
    if "fric_k" not in pred_params and "fric_k_final" in pred_params:
        pred_params["fric_k"] = pred_params["fric_k_final"]

    gaussians = Gaussian(
        sh_degree=0,
        aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
        mininum_kernel_size=0.0009,
        scaling_bias=0.004,
        opacity_bias=0.1,
        scaling_activation="softplus",
    )
    gaussians.load_ply(gs_path_obj)
    xyz_all = gaussians._xyz

    xyz = uniform_sampling(xyz_all, voxel_size=0.01)
    xyz = xyz[random.sample(range(xyz.shape[0]), 2048), :]

    simulator = setup_simulator_from_prediction(cfg, xyz.squeeze(0))
    simulator.set_all_particle(xyz_all)
    simulator.set_physics_params(pred_params)

    v = simulator.init_v.detach().clone()

    sim_xyz_seq = []
    all_view_videos = []
    for frame_id in range(num_frames):
        if frame_id == 0:
            rgbs = render_utils.render_snapshot_with_grad_oneview(
                gaussians, offset=(0, 0), r=2, fov=40
            )["color"]
            all_view_videos.append(rgbs)
            sim_xyz_seq.append(xyz_all.detach().clone())
        else:
            is_nan = True
            while is_nan:
                xyz_all_o, xyz_o, v_o, is_nan = simulator(xyz_all, xyz, v, frame_id)
                if is_nan:
                    simulator.n_step = simulator.n_step + cfg.DYNAMIC.STEP_ADD
                else:
                    gaussians._xyz = xyz_all_o
                    xyz = xyz_o.detach().clone()
                    v = v_o.detach().clone()
                    xyz_all = xyz_all_o.detach().clone()
                    rgbs = render_utils.render_snapshot_with_grad_oneview(
                        gaussians, offset=(0, 0), r=2, fov=40
                    )["color"]
                    all_view_videos.append(rgbs)
                    sim_xyz_seq.append(xyz_all.detach().clone())

    all_view_videos = torch.stack(all_view_videos, dim=1)  # [b, T, 3, H, W]
    rendered_video = all_view_videos[0].permute(1, 0, 2, 3).unsqueeze(0).float()  # [1,3,T,H,W]
    target_video = video_tensor[:, :, :num_frames, :, :]

    return rendered_video, target_video, pred_params, sim_xyz_seq


def _to_scalar(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach()
        if x.numel() == 0:
            return None
        return float(x.mean().cpu().item())
    try:
        return float(x)
    except Exception:
        return None


def _to_list(x):
    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().reshape(-1)
        return [float(v.item()) for v in x]
    try:
        return [float(x)]
    except Exception:
        return []


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        video_dir,
        num_frames=30,
        resize=(512, 512),
        dataset_type="test",
        train_obj_count=450,
        train_samples=10,
        test_samples=1,
    ):
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.resize = resize

        obj_names = sorted([d for d in os.listdir(self.video_dir) if os.path.isdir(os.path.join(self.video_dir, d))])
        if dataset_type == "train":
            obj_names = obj_names[:train_obj_count]
            num_samples = train_samples
            num_views = 4
        else:
            obj_names = obj_names[train_obj_count:]
            if len(obj_names) == 0:
                obj_names = sorted(
                    [d for d in os.listdir(self.video_dir) if os.path.isdir(os.path.join(self.video_dir, d))]
                )[-50:]
            num_samples = test_samples
            num_views = 1

        videos = []
        for obj in obj_names:
            for i in range(num_samples):
                rel_dir = f"{obj}/sample_{i:05d}"
                for v in range(num_views):
                    mp4_rel = f"{rel_dir}/view{v}.mp4"
                    mp4_abs = os.path.join(self.video_dir, mp4_rel)
                    if os.path.isfile(mp4_abs):
                        videos.append(mp4_rel)

        self.items = videos

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.items[idx])
        video_tensor = load_video_as_tensor(video_path, num_frames=self.num_frames, resize=self.resize)
        return video_tensor, self.items[idx]


def build_model(model_name: str, internvit_model_name: str, device: torch.device):
    if model_name == "internvit300m":
        model = VideoPhysicsPredictor_internvit300m(internvit_model_name=internvit_model_name).to(device)
    elif model_name == "token2point":
        model = VideoPhysicsPredictor_Token2Point(internvit_model_name=internvit_model_name).to(device)
    elif model_name == "internvit300m_temporalattn":
        model = VideoPhysicsPredictor_internvit300m_temporalattn(
            internvit_model_name=internvit_model_name
        ).to(device)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return model


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


def main():
    parser = argparse.ArgumentParser("Standalone source-aligned test script")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./test_logs")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--resize", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--cd_points", type=int, default=4096)
    parser.add_argument("--mode", type=str, default="video", choices=["output", "video", "metrics"])

    parser.add_argument("--dataset_root", type=str, default="datasets/multiphys_obj500_hash")
    parser.add_argument("--train_obj_count", type=int, default=450)
    parser.add_argument("--train_samples", type=int, default=10)
    parser.add_argument("--test_samples", type=int, default=1)

    parser.add_argument("--model_name", type=str, default="internvit300m_temporalattn",
                        choices=["internvit300m", "internvit300m_temporalattn", "token2point"])
    parser.add_argument(
        "--internvit_model_name",
        type=str,
        default="OpenGVLab/InternViT-300M-448px-V2_5",
        help="HF repo id or local model directory for InternViT backbone",
    )

    parser.add_argument("--cfg_default", type=str, default="")
    parser.add_argument("--cfg_scene", type=str, default="")

    args = parser.parse_args()

    dataset_candidates = [
        Path.cwd(),
        THIS_DIR,
        THIS_DIR / "datasets",
    ]
    cfg_candidates = [
        Path.cwd(),
        THIS_DIR,
        THIS_DIR / "config",
        THIS_DIR / "config" / "mpm_synthetic",
        THIS_DIR / "configs",
        THIS_DIR / "configs" / "mpm_synthetic",
    ]

    args.dataset_root = _resolve_existing_path(args.dataset_root, dataset_candidates, expect="dir")

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "video"), exist_ok=True)
    per_video_csv = os.path.join(args.outdir, "per_video_params.csv")
    per_video_jsonl = os.path.join(args.outdir, "per_video_params.jsonl")

    if not os.path.isfile(per_video_csv):
        with open(per_video_csv, "w") as f:
            f.write(
                "video_file,k,fric_k,m,damp,err_k,err_fric_k,err_m,err_damp,"
                "rec_loss,pred_loss,psnr_rec,ssim_rec,lpips_rec,psnr_pred,ssim_pred,"
                "lpips_pred,cd_rec,emd_rec,cd_pred,emd_pred\n"
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    predictor = build_model(args.model_name, args.internvit_model_name, device)
    load_checkpoint_weights(args.ckpt, predictor, device)
    predictor.eval()

    cfg = None
    if args.mode in ["video", "metrics"]:
        if not args.cfg_default or not args.cfg_scene:
            raise ValueError(
                "--cfg_default and --cfg_scene are required when mode is video/metrics"
            )
        args.cfg_default = _resolve_existing_path(args.cfg_default, cfg_candidates, expect="file")
        args.cfg_scene = _resolve_existing_path(args.cfg_scene, cfg_candidates, expect="file")
        cfg = CN(new_allowed=True)
        cfg.merge_from_file(args.cfg_default)
        cfg.merge_from_file(args.cfg_scene)

    testset = Dataset(
        video_dir=args.dataset_root,
        num_frames=args.frames,
        resize=tuple(args.resize),
        dataset_type="test",
        train_obj_count=args.train_obj_count,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

    evaluator = VideoEvaluator()
    loss_func = nn.SmoothL1Loss(reduction="mean")

    rec_total_loss = rec_total_psnr = rec_total_ssim = rec_total_lpips = rec_total_chamfer = 0.0
    pred_total_loss = pred_total_psnr = pred_total_ssim = pred_total_lpips = pred_total_chamfer = 0.0
    rec_total_emd = pred_total_emd = 0.0
    num_rec_videos = num_pred_videos = num_videos = 0

    param_err_sum = {"k": 0.0, "fric_k": 0.0, "m": 0.0, "damp": 0.0}
    pred_k, pred_fric, pred_m, pred_damp = [], [], [], []

    for video_tensor, video_file in testloader:
        video_tensor = video_tensor.to(device, non_blocking=True)
        parts = video_file[0].split("/")
        obj_name, sample_name = parts[0], parts[1]
        gs_path_obj = os.path.join(testset.video_dir, obj_name, "gaussian.ply")

        if args.mode == "output":
            rec_frame_startend = [0, 20]
            predictor_input = video_tensor[:, :, rec_frame_startend[0] : rec_frame_startend[1], :, :]
            with torch.no_grad():
                pred_params = predictor(predictor_input)
            rendered_video = target_video = sim_xyz_seq = None
        else:
            with torch.no_grad():
                rendered_video, target_video, pred_params, sim_xyz_seq = infer_one_video(
                    video_tensor, predictor, cfg, gs_path_obj, num_frames=args.frames
                )

        pred_item = {
            "video_file": video_file[0],
            "k": _to_scalar(pred_params.get("k", None)),
            "fric_k": _to_scalar(pred_params.get("fric_k", None)),
            "m": _to_scalar(pred_params.get("m", None)),
            "damp": _to_scalar(pred_params.get("damp", None)),
        }

        pred_k.extend(_to_list(pred_params.get("k", None)))
        pred_fric.extend(_to_list(pred_params.get("fric_k", None)))
        pred_m.extend(_to_list(pred_params.get("m", None)))
        pred_damp.extend(_to_list(pred_params.get("damp", None)))

        sample_dir = os.path.join(testset.video_dir, obj_name, sample_name)
        pkl_path = os.path.join(sample_dir, "physics_params.pkl")
        errs = {}
        if args.mode == "metrics" and os.path.isfile(pkl_path):
            with open(pkl_path, "rb") as f:
                gt_params = pickle.load(f)
            errs_raw = compute_param_l1_errors(pred_params, gt_params)
            for key in param_err_sum:
                if key in errs_raw:
                    val = float(errs_raw[key])
                    errs[key] = val
                    param_err_sum[key] += val

        err_item = {
            "err_k": errs.get("k", None),
            "err_fric_k": errs.get("fric_k", None),
            "err_m": errs.get("m", None),
            "err_damp": errs.get("damp", None),
        }

        if args.mode in ["video", "metrics"]:
            out_video_path = os.path.join(args.outdir, "video", f"{obj_name}_{sample_name}_pred.mp4")
            rendered_video_np = rendered_video[0].permute(1, 2, 3, 0).detach().cpu().numpy()
            target_video_np = target_video[0].permute(1, 2, 3, 0).detach().cpu().numpy()
            concat_video = np.concatenate([target_video_np, rendered_video_np], axis=2)
            imageio.mimwrite(out_video_path, (np.clip(concat_video, 0.0, 1.0) * 255.0).astype(np.uint8), fps=30)

        if args.mode == "metrics":
            mask = torch.zeros_like(target_video)
            lower_bound = torch.tensor([0.0, 0.0, 0.0], device=device).view(1, 3, 1, 1, 1)
            upper_bound = torch.tensor([0.0392, 0.0392, 0.0392], device=device).view(1, 3, 1, 1, 1)
            mask_cond = ((target_video >= lower_bound) & (target_video <= upper_bound)).all(dim=1, keepdim=True)
            mask = torch.where(mask_cond, torch.zeros_like(mask), torch.ones_like(mask))
            rendered_video = rendered_video * mask

        rec_t0, rec_t1 = 0, min(20, args.frames)
        pred_t0, pred_t1 = min(20, args.frames), min(30, args.frames)

        if args.mode == "metrics":
            if rec_t1 > rec_t0:
                loss_rec = float(
                    loss_func(rendered_video[:, :, rec_t0:rec_t1], target_video[:, :, rec_t0:rec_t1]).item()
                )
                rec_total_loss += loss_rec
                num_rec_videos += 1
            else:
                loss_rec = 0.0

            if pred_t1 > pred_t0:
                loss_pred = float(
                    loss_func(rendered_video[:, :, pred_t0:pred_t1], target_video[:, :, pred_t0:pred_t1]).item()
                )
                pred_total_loss += loss_pred
                num_pred_videos += 1
            else:
                loss_pred = 0.0
        else:
            loss_rec = loss_pred = None

        if args.mode == "metrics":
            rv = rendered_video[0].permute(1, 2, 3, 0).detach().cpu().numpy()
            tv = target_video[0].permute(1, 2, 3, 0).detach().cpu().numpy()
            rv_uint8 = (np.clip(rv, 0.0, 1.0) * 255.0).astype(np.uint8)
            tv_uint8 = (np.clip(tv, 0.0, 1.0) * 255.0).astype(np.uint8)

            if rec_t1 > rec_t0:
                psnr_r, ssim_r, lpips_r = evaluator.cal_one_video_psnr_ssim_lpips(
                    [frame for frame in tv_uint8[rec_t0:rec_t1]],
                    [frame for frame in rv_uint8[rec_t0:rec_t1]],
                )
                rec_total_psnr += psnr_r
                rec_total_ssim += ssim_r
                rec_total_lpips += lpips_r
            else:
                psnr_r = ssim_r = lpips_r = 0.0

            if pred_t1 > pred_t0:
                psnr_p, ssim_p, lpips_p = evaluator.cal_one_video_psnr_ssim_lpips(
                    [frame for frame in tv_uint8[pred_t0:pred_t1]],
                    [frame for frame in rv_uint8[pred_t0:pred_t1]],
                )
                pred_total_psnr += psnr_p
                pred_total_ssim += ssim_p
                pred_total_lpips += lpips_p
            else:
                psnr_p = ssim_p = lpips_p = 0.0
        else:
            psnr_r = ssim_r = lpips_r = None
            psnr_p = ssim_p = lpips_p = None

        if args.mode == "metrics":
            states_dir = os.path.join(testset.video_dir, obj_name, sample_name, "states")
            state_files = sorted([f for f in os.listdir(states_dir) if f.endswith(".pt")])

            cd_sum, valid = 0.0, 0
            emd_sum, emd_valid = 0.0, 0
            for frame_idx in range(rec_t0, rec_t1):
                state_path = os.path.join(states_dir, state_files[frame_idx])
                state = torch.load(state_path, map_location="cpu", weights_only=True)
                gt_pc = torch.as_tensor(state["xyz_all"], dtype=torch.float32).to(device, non_blocking=True)
                sim_pc = sim_xyz_seq[frame_idx].to(device, non_blocking=True)
                gt_pc = gt_pc[torch.isfinite(gt_pc).all(dim=-1)]
                sim_pc = sim_pc[torch.isfinite(sim_pc).all(dim=-1)]
                if gt_pc.shape[0] == 0 or sim_pc.shape[0] == 0:
                    continue
                cd_t = chamfer_distance(subsample_points(sim_pc, args.cd_points), subsample_points(gt_pc, args.cd_points), squared=True)
                cd_sum += float(cd_t.item())
                valid += 1
                m = min(min(args.cd_points, 512), gt_pc.shape[0], sim_pc.shape[0])
                if m >= 4:
                    gt_e = subsample_points(gt_pc, m)
                    sim_e = subsample_points(sim_pc, m)
                    m_eq = min(gt_e.shape[0], sim_e.shape[0])
                    gt_e = subsample_points(gt_e, m_eq)
                    sim_e = subsample_points(sim_e, m_eq)
                    if gt_e.shape[0] == sim_e.shape[0] and gt_e.shape[0] > 0:
                        emd_t = earth_movers_distance(gt_e, sim_e, squared=True)
                        emd_sum += float(emd_t.item())
                        emd_valid += 1
            cd_rec = (cd_sum / valid) if valid > 0 else 0.0
            emd_rec = (emd_sum / emd_valid) if emd_valid > 0 else 0.0
            rec_total_chamfer += cd_rec
            rec_total_emd += emd_rec

            cd_sum, valid = 0.0, 0
            emd_sum, emd_valid = 0.0, 0
            for frame_idx in range(pred_t0, pred_t1):
                state_path = os.path.join(states_dir, state_files[frame_idx])
                state = torch.load(state_path, map_location="cpu", weights_only=True)
                gt_pc = torch.as_tensor(state["xyz_all"], dtype=torch.float32).to(device, non_blocking=True)
                sim_pc = sim_xyz_seq[frame_idx].to(device, non_blocking=True)
                gt_pc = gt_pc[torch.isfinite(gt_pc).all(dim=-1)]
                sim_pc = sim_pc[torch.isfinite(sim_pc).all(dim=-1)]
                if gt_pc.shape[0] == 0 or sim_pc.shape[0] == 0:
                    continue
                cd_t = chamfer_distance(subsample_points(sim_pc, args.cd_points), subsample_points(gt_pc, args.cd_points), squared=True)
                cd_sum += float(cd_t.item())
                valid += 1
                m = min(min(args.cd_points, 512), gt_pc.shape[0], sim_pc.shape[0])
                if m >= 4:
                    gt_e = subsample_points(gt_pc, m)
                    sim_e = subsample_points(sim_pc, m)
                    m_eq = min(gt_e.shape[0], sim_e.shape[0])
                    gt_e = subsample_points(gt_e, m_eq)
                    sim_e = subsample_points(sim_e, m_eq)
                    if gt_e.shape[0] == sim_e.shape[0] and gt_e.shape[0] > 0:
                        emd_t = earth_movers_distance(gt_e, sim_e, squared=True)
                        emd_sum += float(emd_t.item())
                        emd_valid += 1
            cd_pred = (cd_sum / valid) if valid > 0 else 0.0
            emd_pred = (emd_sum / emd_valid) if emd_valid > 0 else 0.0
            pred_total_chamfer += cd_pred
            pred_total_emd += emd_pred
        else:
            cd_rec = emd_rec = cd_pred = emd_pred = None

        per_video_row = [
            pred_item["video_file"],
            pred_item["k"],
            pred_item["fric_k"],
            pred_item["m"],
            pred_item["damp"],
            err_item["err_k"],
            err_item["err_fric_k"],
            err_item["err_m"],
            err_item["err_damp"],
            loss_rec,
            loss_pred,
            psnr_r,
            ssim_r,
            lpips_r,
            psnr_p,
            ssim_p,
            lpips_p,
            cd_rec,
            emd_rec,
            cd_pred,
            emd_pred,
        ]
        with open(per_video_csv, "a") as fcsv:
            fcsv.write(",".join([("" if v is None else str(v)) for v in per_video_row]) + "\n")

        per_video_record = {
            **pred_item,
            **err_item,
            "rec_loss": loss_rec,
            "pred_loss": loss_pred,
            "psnr_rec": psnr_r,
            "ssim_rec": ssim_r,
            "lpips_rec": lpips_r,
            "psnr_pred": psnr_p,
            "ssim_pred": ssim_p,
            "lpips_pred": lpips_p,
            "cd_rec": cd_rec,
            "emd_rec": emd_rec,
            "cd_pred": cd_pred,
            "emd_pred": emd_pred,
        }
        with open(per_video_jsonl, "a") as fj:
            fj.write(json.dumps(per_video_record) + "\n")

        num_videos += 1
        if args.mode == "metrics":
            rec_line = (
                f"[REC]  {video_file[0]} | loss={loss_rec:.6f}, PSNR={psnr_r:.4f}, "
                f"SSIM={ssim_r:.4f}, LPIPS={lpips_r:.4f}, CD={cd_rec:.6f}, EMD={emd_rec:.6f}"
            )
            pred_line = (
                f"[PRED] {video_file[0]} | loss={loss_pred:.6f}, PSNR={psnr_p:.4f}, "
                f"SSIM={ssim_p:.4f}, LPIPS={lpips_p:.4f}, CD={cd_pred:.6f}, EMD={emd_pred:.6f}"
            )
            print(rec_line)
            print(pred_line)
            with open(os.path.join(args.outdir, "result_logs.txt"), "a") as f:
                f.write(rec_line + "\n")
                f.write(pred_line + "\n")
        else:
            print(f"[Done] {video_file[0]}")

    if args.mode == "metrics":
        rec_avg_loss = rec_total_loss / max(1, num_rec_videos)
        rec_avg_psnr = rec_total_psnr / max(1, num_rec_videos)
        rec_avg_ssim = rec_total_ssim / max(1, num_rec_videos)
        rec_avg_lpips = rec_total_lpips / max(1, num_rec_videos)
        rec_avg_chamfer = rec_total_chamfer / max(1, num_rec_videos)
        rec_avg_emd = rec_total_emd / max(1, num_rec_videos)

        pred_avg_loss = pred_total_loss / max(1, num_pred_videos)
        pred_avg_psnr = pred_total_psnr / max(1, num_pred_videos)
        pred_avg_ssim = pred_total_ssim / max(1, num_pred_videos)
        pred_avg_lpips = pred_total_lpips / max(1, num_pred_videos)
        pred_avg_chamfer = pred_total_chamfer / max(1, num_pred_videos)
        pred_avg_emd = pred_total_emd / max(1, num_pred_videos)

        avg_errs = {k: (param_err_sum[k] / max(1, num_videos)) for k in param_err_sum}

        print("\n[Test Summary]")
        print(f"- Videos: {num_videos}")
        print(
            f"- Reconstruction [0,20): Loss={rec_avg_loss:.6f}, PSNR={rec_avg_psnr:.4f}, "
            f"SSIM={rec_avg_ssim:.4f}, LPIPS={rec_avg_lpips:.4f}, "
            f"Chamfer={rec_avg_chamfer:.6f}, EMD={rec_avg_emd:.6f}"
        )
        print(
            f"- Prediction    [20,30): Loss={pred_avg_loss:.6f}, PSNR={pred_avg_psnr:.4f}, "
            f"SSIM={pred_avg_ssim:.4f}, LPIPS={pred_avg_lpips:.4f}, "
            f"Chamfer={pred_avg_chamfer:.6f}, EMD={pred_avg_emd:.6f}"
        )
        print(
            f"- Param L1: k={avg_errs['k']:.4f}, fric_k={avg_errs['fric_k']:.4f}, "
            f"m={avg_errs['m']:.4f}, damp={avg_errs['damp']:.4f}"
        )

        summary = {
            "num_videos": num_videos,
            "reconstruction": {
                "loss": rec_avg_loss,
                "psnr": rec_avg_psnr,
                "ssim": rec_avg_ssim,
                "lpips": rec_avg_lpips,
                "chamfer": rec_avg_chamfer,
                "emd": rec_avg_emd,
            },
            "prediction": {
                "loss": pred_avg_loss,
                "psnr": pred_avg_psnr,
                "ssim": pred_avg_ssim,
                "lpips": pred_avg_lpips,
                "chamfer": pred_avg_chamfer,
                "emd": pred_avg_emd,
            },
            "param_err": avg_errs,
            "pred_stats": {
                "k": {"mean": float(np.mean(pred_k)) if pred_k else None, "std": float(np.std(pred_k)) if pred_k else None},
                "fric_k": {
                    "mean": float(np.mean(pred_fric)) if pred_fric else None,
                    "std": float(np.std(pred_fric)) if pred_fric else None,
                },
                "m": {"mean": float(np.mean(pred_m)) if pred_m else None, "std": float(np.std(pred_m)) if pred_m else None},
                "damp": {
                    "mean": float(np.mean(pred_damp)) if pred_damp else None,
                    "std": float(np.std(pred_damp)) if pred_damp else None,
                },
            },
        }
        with open(os.path.join(args.outdir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {args.outdir}")


if __name__ == "__main__":
    main()
