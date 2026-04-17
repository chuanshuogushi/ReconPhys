# Keyboard control for two control points with step-by-step simulation and rendering.
import os
os.environ['SPCONV_ALGO'] = 'native'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import math
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import imageio
from PIL import Image
import pygame
from yacs.config import CfgNode as CN

THIS_DIR = Path(__file__).resolve().parent
from sms_lib.utils.transform import uniform_sampling
from trellis.utils import render_utils
from trellis.representations.gaussian.gaussian_model import Gaussian

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


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_device(v, device) for v in obj)
    return obj


def load_physics_params(params_path: str, device: torch.device):
    loaded = torch.load(params_path, map_location=device)
    return _to_device(loaded, device)

# Highlight only one nearest Gaussian per control point.
def highlight_controller_points_on_gaussians(gaussians, simulator, opacity=1.0, scale_factor=3.0, colors=None):
    """
    Highlight only the nearest Gaussian for each control point,
    instead of the full affected region.
    Color assignment is stable by control-point index.
    """
    if (not getattr(simulator, 'has_controller', False)) or simulator.n_ctrl == 0:
        return None

    n_ctrl = int(simulator.n_ctrl)
    if colors is not None:
        palette = [tuple(map(float, c)) for c in colors]
    else:
        palette = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)] if n_ctrl >= 2 else [(1.0, 0.0, 0.0)]
    while len(palette) < n_ctrl:
        palette.append(palette[-1])

    xyz_all = gaussians._xyz  # [N,3]
    # Control-point positions (usually tracked in the simulator).
    ctrl_pos = simulator.ctrl_xyz if hasattr(simulator, 'ctrl_xyz') else None
    if ctrl_pos is None:
        # Fallback: approximate with the centroid of each binding set.
        ctrl_pos = []
        for c in range(n_ctrl):
            bind_idx_c = simulator.ctrl_bind_index[c].to(xyz_all.device)
            if bind_idx_c.numel() == 0:
                # Skip if no bindings are available.
                ctrl_pos.append(torch.zeros(3, device=xyz_all.device, dtype=xyz_all.dtype))
            else:
                ctrl_pos.append(xyz_all[bind_idx_c].mean(dim=0))
        ctrl_pos = torch.stack(ctrl_pos, dim=0)  # [n_ctrl,3]

    # Select one nearest Gaussian index per control point.
    # Prefer searching inside each binding set for better stability.
    chosen_idx = []
    for c in range(n_ctrl):
        pos = ctrl_pos[c].view(1, 3)  # [1,3]
        # Prefer nearest inside binding set to avoid crossing regions.
        bind_idx_c = simulator.ctrl_bind_index[c].to(xyz_all.device)
        if bind_idx_c.numel() > 0:
            pts = xyz_all[bind_idx_c]  # [Kc,3]
            d2 = torch.sum((pts - pos) ** 2, dim=1)  # [Kc]
            nn_local = torch.argmin(d2)
            chosen_idx.append(bind_idx_c[nn_local].item())
        else:
            # If no binding set exists, use global nearest.
            d2 = torch.sum((xyz_all - pos) ** 2, dim=1)
            chosen_idx.append(torch.argmin(d2).item())

    chosen_idx = torch.tensor(chosen_idx, device=xyz_all.device, dtype=torch.long).unique()

    if chosen_idx.numel() == 0:
        return None

    # Backup original attributes.
    backup = {
        'idx': chosen_idx,
        'features_dc': gaussians._features_dc[chosen_idx].clone(),
        'opacity': gaussians._opacity[chosen_idx].clone(),
        'scaling': gaussians._scaling[chosen_idx].clone(),
    }

    # Increase opacity and scale.
    op = gaussians._opacity
    op_val = torch.tensor(5.0, dtype=op.dtype, device=op.device).view(1, 1)
    op[chosen_idx] = op_val

    sc = gaussians._scaling
    sc[chosen_idx] = sc[chosen_idx] + math.log(scale_factor)

    # Stable coloring by control-point index.
    f = gaussians._features_dc
    for c in range(n_ctrl):
        target = torch.tensor(palette[c], dtype=f.dtype, device=f.device).view(1, 1, 3)
        # If no selected point belongs to this control point, skip.
        bind_idx_c = simulator.ctrl_bind_index[c].to(xyz_all.device)
        # Color one point in chosen_idx ∩ bind_idx_c.
        inter_mask = torch.isin(chosen_idx, bind_idx_c)
        if inter_mask.any():
            idx_c = chosen_idx[inter_mask]
            # If multiple, pick one (usually one per control point).
            f[idx_c[:1]] = target * 2.0

    return backup

def restore_gaussians_from_backup(gaussians, backup):
    if backup is None:
        return
    idx = backup['idx']
    gaussians._features_dc[idx] = backup['features_dc']
    gaussians._opacity[idx] = backup['opacity']
    gaussians._scaling[idx] = backup['scaling']

# ----------------- Highlight controller-affected Gaussians (stable ownership) -----------------
def highlight_controllers_on_gaussians(gaussians, simulator, opacity=1.0, scale_factor=3.0, colors=None):
    """
    Color and enlarge Gaussians affected by control points.
    Call restore after rendering.
    Ownership is assigned stably by control-point index order.
    """
    if (not getattr(simulator, 'has_controller', False)) or simulator.n_ctrl == 0:
        return None

    n_ctrl = int(simulator.n_ctrl)
    if colors is not None:
        palette = [tuple(map(float, c)) for c in colors]
    else:
        palette = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)] if n_ctrl >= 2 else [(1.0, 0.0, 0.0)]
    while len(palette) < n_ctrl:
        palette.append(palette[-1])

    intrp_index = simulator.intrp_index  # [N_all, k_binding]
    union_mask = torch.zeros(intrp_index.shape[0], dtype=torch.bool, device=intrp_index.device)
    ctrl_masks = []
    for c in range(n_ctrl):
        bind_idx_c = simulator.ctrl_bind_index[c].to(intrp_index.device)  # [Kc]
        mask_c = torch.isin(intrp_index, bind_idx_c).any(dim=1)           # [N_all]
        ctrl_masks.append(mask_c)
        union_mask |= mask_c
    gauss_idx_union = torch.nonzero(union_mask, as_tuple=False).squeeze(1)
    if gauss_idx_union.numel() == 0:
        return None

    backup = {
        'idx': gauss_idx_union,
        'features_dc': gaussians._features_dc[gauss_idx_union].clone(),
        'opacity': gaussians._opacity[gauss_idx_union].clone(),
        'scaling': gaussians._scaling[gauss_idx_union].clone(),
    }

    op = gaussians._opacity
    op_val = torch.tensor(5.0, dtype=op.dtype, device=op.device).view(1, 1)
    op[gauss_idx_union] = op_val

    sc = gaussians._scaling
    sc[gauss_idx_union] = sc[gauss_idx_union] + math.log(scale_factor)

    f = gaussians._features_dc  # [N, 1, 3]
    owned = torch.zeros(gauss_idx_union.shape[0], dtype=torch.bool, device=intrp_index.device)
    for c in range(n_ctrl):
        mask_c = ctrl_masks[c][gauss_idx_union]
        assign_mask = (~owned) & mask_c
        if assign_mask.any():
            sub_idx = gauss_idx_union[assign_mask]
            target = torch.tensor(palette[c], dtype=f.dtype, device=f.device).view(1, 1, 3)
            f[sub_idx] = target * 2.0
            owned[assign_mask] = True

    return backup

# ----------------- Physics and simulator setup -----------------
def setup_simulator_from_prediction(cfg, xyz):
    from sms_lib.utils.builder import build_simulator
    # from sms_lib.models.spring_mass.Spring_Mass import Spring_Mass
    from sms_lib.models.spring_mass.Spring_Mass_simplify import Spring_Mass
    cfg.DAMPING = True
    simulator = build_simulator(
        cfg.DYNAMIC,
        xyz=xyz,
        data=cfg.DATA,
        init_velocity=[0, 0, 0],
        # load_g=[0, 0, -9.8]
        load_g = [0,0,0]
    )
    return simulator.cuda()

# ----------------- Utility: initialize two control points (left/right edges) -----------------
def pick_two_controllers_from_xyz_all(xyz_all: torch.Tensor):
    """
    Use the points with minimum and maximum x as initial control points.
    Returns torch.tensor([2,3]) on the same device/dtype as xyz_all.
    """
    xyz_all_cpu = xyz_all.detach().cpu()
    x_coords = xyz_all_cpu[:, 0]
    left_idx = torch.argmin(x_coords).item()
    right_idx = torch.argmax(x_coords).item()
    left_pos = xyz_all_cpu[left_idx]
    right_pos = xyz_all_cpu[right_idx]
    ctrl = torch.stack([left_pos, right_pos], dim=0)  # [2,3]
    
    # Print initial control-point distribution on the object.
    print(f'Initial control points: left {left_pos.numpy()}, right {right_pos.numpy()}')
    print(f'Object range: x[{x_coords.min().item()}, {x_coords.max().item()}], y[{xyz_all_cpu[:,1].min().item()}, {xyz_all_cpu[:,1].max().item()}], z[{xyz_all_cpu[:,2].min().item()}, {xyz_all_cpu[:,2].max().item()}]')
    print(f'Object center: {xyz_all_cpu.mean(dim=0).numpy()}')
    return ctrl.to(xyz_all.device, dtype=xyz_all.dtype)

# ----------------- Keyboard mapping -----------------
def build_keymap(step=0.06):
    """
        Returns: dict(pygame_key -> (ctrl_id, delta_xyz))
        Convention:
            - ctrl0: wasd front/back/left/right (y+/y- ; x-/x+), q/e down/up (z-/z+)
            - ctrl1: ijkl front/back/left/right (y+/y- ; x-/x+), o/u down/up (z-/z+)
    """
    s = float(step)
    key = pygame.key
    # K = pygame.K_

    m = {}
    # ctrl 0
    m[pygame.K_w] = (0, np.array([0.0, +s, 0.0], dtype=np.float32))
    m[pygame.K_s] = (0, np.array([0.0, -s, 0.0], dtype=np.float32))
    m[pygame.K_a] = (0, np.array([-s, 0.0, 0.0], dtype=np.float32))
    m[pygame.K_d] = (0, np.array([+s, 0.0, 0.0], dtype=np.float32))
    m[pygame.K_q] = (0, np.array([0.0, 0.0, -s], dtype=np.float32))
    m[pygame.K_e] = (0, np.array([0.0, 0.0, +s], dtype=np.float32))
    # ctrl 1
    m[pygame.K_i] = (1, np.array([0.0, +s, 0.0], dtype=np.float32))
    m[pygame.K_k] = (1, np.array([0.0, -s, 0.0], dtype=np.float32))
    m[pygame.K_j] = (1, np.array([-s, 0.0, 0.0], dtype=np.float32))
    m[pygame.K_l] = (1, np.array([+s, 0.0, 0.0], dtype=np.float32))
    m[pygame.K_u] = (1, np.array([0.0, 0.0, -s], dtype=np.float32))
    m[pygame.K_o] = (1, np.array([0.0, 0.0, +s], dtype=np.float32))
    return m

def put_on_ground_xyz_all(xyz_all: torch.Tensor):
    """
    Translate xyz_all so its lowest point is exactly on the ground.
    return: (xyz_all_new, delta)
    """
    ground_axis, ground = 2, -0.2

    xyz_all_new = xyz_all.clone()
    min_h = xyz_all_new[:, ground_axis].min()
    delta = ground - min_h

    xyz_all_new[:, ground_axis] = xyz_all_new[:, ground_axis] + delta
    return xyz_all_new, float(delta)

def replace_black_bg(img_rgb: np.ndarray, bg_rgb: np.ndarray, thr: int = 8) -> np.ndarray:
    """
    Build alpha from pure/near-black detection and replace black pixels with background.
    - img_rgb: (H,W,3) uint8, RGB (render output)
    - bg_rgb:  (H,W,3) uint8, RGB (background image, same size required)
    - thr:     black threshold, pixel is background if all channels <= thr
    return:    (H,W,3) uint8, RGB
    """
    assert img_rgb.dtype == np.uint8 and bg_rgb.dtype == np.uint8
    assert img_rgb.shape == bg_rgb.shape and img_rgb.ndim == 3 and img_rgb.shape[2] == 3

    # alpha=1 means foreground; alpha=0 means black background.
    is_black = (img_rgb[..., 0] <= thr) & (img_rgb[..., 1] <= thr) & (img_rgb[..., 2] <= thr)
    alpha = (~is_black).astype(np.float32)[..., None]  # (H,W,1)

    out = img_rgb.astype(np.float32) * alpha + bg_rgb.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)

# ----------------- Main -----------------
def build_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gs_path', type=str, required=True, help='path to gaussian.ply')
    p.add_argument('--cfg_default', type=str, default='default.yaml')
    p.add_argument('--cfg_multiscene', type=str, default='multiscene.yaml')
    p.add_argument('--bg_image', type=str, default='', help='optional background image path; use black if empty')
    p.add_argument('--params_path', type=str, required=True, help='path to saved predicted physics params (.pt)')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--step', type=float, default=0.02, help='movement step per key press (world coordinates)')
    p.add_argument('--repeat_hz', type=int, default=15, help='key-hold repeat frequency in Hz (pygame set_repeat)')
    p.add_argument('--win', type=str, default='TRELLIS Control', help='window title')
    p.add_argument('--fps_cap', type=int, default=60, help='max main-loop FPS cap')
    p.add_argument('--max_frame', type=int, default=1_000_000, help='max simulation steps (safety limit)')
    p.add_argument('--save_dir', type=str, default='outputs', help='optional directory to save per-step outputs')
    p.add_argument('--cuda', type=int, default=-1, help='set CUDA_VISIBLE_DEVICES (-1 keeps current env)')
    return p.parse_args()


def main():
    args = build_args()

    file_candidates = [
        Path.cwd(),
        THIS_DIR,
        THIS_DIR / "config" / "mpm_synthetic",
        THIS_DIR / "demos",
    ]
    args.gs_path = _resolve_existing_path(args.gs_path, file_candidates, expect="file")
    args.cfg_default = _resolve_existing_path(args.cfg_default, file_candidates, expect="file")
    args.cfg_multiscene = _resolve_existing_path(args.cfg_multiscene, file_candidates, expect="file")
    args.params_path = _resolve_existing_path(args.params_path, file_candidates, expect="file")
    if args.bg_image:
        args.bg_image = _resolve_existing_path(args.bg_image, file_candidates, expect="file")

    if args.cuda >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)

    # Random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)

    # Load config
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.cfg_default)
    cfg.merge_from_file(args.cfg_multiscene)

    # Load Gaussian
    gaussians = Gaussian(
        sh_degree=0,
        aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
        mininum_kernel_size=0.0009,
        scaling_bias=0.004,
        opacity_bias=0.1,
        scaling_activation='softplus',
    )
    gaussians.load_ply(args.gs_path)
    # ===== Put Gaussian xyz_all on the ground before sparse sampling and simulator build =====
    gaussians._xyz, delta_g = put_on_ground_xyz_all(gaussians._xyz)
    print(f"[put_on_ground] shifted along ground axis by delta={delta_g:.6f}")


    xyz_all = gaussians._xyz  # [N,3], torch.cuda.FloatTensor

    # Select sparse points (2048) for simulator construction
    try:
        xyz_sparse = uniform_sampling(xyz_all, voxel_size=0.01)
        n_pick = min(2048, xyz_sparse.shape[0])
        xyz_sparse = xyz_sparse[random.sample(range(xyz_sparse.shape[0]), n_pick), :]
    except Exception:
        xyz_sparse = uniform_sampling(xyz_all, voxel_size=0.001)
        n_pick = min(2048, xyz_sparse.shape[0])
        xyz_sparse = xyz_sparse[random.sample(range(xyz_sparse.shape[0]), n_pick), :]
    assert n_pick == 2048, f"n_pick should be 2048 but got {n_pick}"

    # Build simulator
    simulator = setup_simulator_from_prediction(cfg, xyz_sparse.squeeze(0))
    simulator.set_all_particle(xyz_all)
    physics_params = load_physics_params(args.params_path, xyz_all.device)
    simulator.set_physics_params(physics_params)
    v = simulator.init_v.detach().clone()
    xyz = xyz_sparse.detach().clone()

    # Save initial snapshots for reset
    init_gaussians_xyz = gaussians._xyz.detach().clone()
    init_xyz_sparse = xyz_sparse.detach().clone()
    init_v = v.detach().clone()

    # Initialize two control points (left and right edges)
    ctrl_curr = pick_two_controllers_from_xyz_all(xyz_all)  # [2,3]
    simulator.init_controller(ctrl_curr)  # Establish binding relation
    ctrl_prev = ctrl_curr.detach().clone()
    init_ctrl = ctrl_curr.detach().clone()

    # Optional save directory
    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Pygame window
    pygame.init()
    height = 1024
    width = 1024
    win_size = (width, height)
    video_frames = []
    screen = pygame.display.set_mode(win_size)
    pygame.display.set_caption(args.win)
    clock = pygame.time.Clock()
    # Key-hold repeat: delay=initial delay(ms), interval=repeat interval(ms)
    if args.repeat_hz > 0:
        interval = max(1, int(1000 / args.repeat_hz))
        pygame.key.set_repeat(1, interval)
    keymap = build_keymap(step=args.step)

    if args.bg_image:
        bg_img = np.array(Image.open(args.bg_image).convert("RGB").resize((width, height)))
    else:
        bg_img = np.zeros((height, width, 3), dtype=np.uint8)
    # Render once at initialization
    def render_and_show():
        # Highlight one Gaussian per control point
        # backup = highlight_controller_points_on_gaussians(
        #     gaussians, simulator, opacity=1.0, scale_factor=100.0,
        #     colors=[(1.0, 0.0, 0.0), (0.0, 0.2, 1.0)]
        # )
        # Highlight affected control regions
        # backup = highlight_controllers_on_gaussians(
        #     gaussians, simulator, opacity=1.0, scale_factor=3.0,
        #     colors=[(1.0, 0.0, 0.0), (0.0, 0.2, 1.0)]
        # )
        # out = render_utils.render_snapshot_downview(gaussians,resolution=1024,r=13,fov=8)
        out = render_utils.render_snapshot_oneview(gaussians,resolution=1024)
        img = out['color'][0]  # [H,W,3], uint8, RGB
        img = replace_black_bg(img, bg_img, thr=8)
        img = np.clip(img, 0, 255).astype(np.uint8)
        video_frames.append(img)
        # Convert to pygame Surface (expects [W,H,3])
        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))  # (W,H,3)
        if surf.get_size() != win_size:
            surf = pygame.transform.smoothscale(surf, win_size)
        screen.blit(surf, (0, 0))

        # Overlay simple key hints
        try:
            font = pygame.font.SysFont(None, 18)
            lines = [
                "Ctrl0: W/S(+/-y) A/D(-/+x) Q/E(-/+z)",
                "Ctrl1: I/K(+/-y) J/L(-/+x) U/O(-/+z)",
                "R: reset, ESC: quit",
            ]
            y = 5
            for t in lines:
                text = font.render(t, True, (255, 255, 255))
                screen.blit(text, (5, y))
                y += 18
        except Exception:
            pass

        pygame.display.flip()

    render_and_show()

    print("Live control started:")
    print("  Control point 0: W/S front-back(+y/-y), A/D left-right(-x/+x), Q/E down-up(-z/+z)")
    print("  Control point 1: I/K front-back(+y/-y), J/L left-right(-x/+x), U/O down-up(-z/+z)")
    print("  R to reset, ESC to quit.")

    # Main loop
    step_id = 0
    running = True
    while running and step_id < args.max_frame:
        # Cap loop FPS (affects event handling and display, not simulation math)
        clock.tick(args.fps_cap)

        # Flags: controller moved this frame / just reset
        moved_this_frame = False
        just_reset = False

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                print(f"Key: {pygame.key.name(event.key)}")
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                # Reset to initial state
                if event.key == pygame.K_r:
                    # Restore Gaussian coordinates
                    gaussians._xyz = init_gaussians_xyz.detach().clone()
                    # Rebuild simulator and restore physics params
                    simulator = setup_simulator_from_prediction(cfg, init_xyz_sparse.squeeze(0))
                    simulator.set_all_particle(gaussians._xyz)
                    simulator.set_physics_params(physics_params)
                    # Restore sparse points and velocity
                    xyz = init_xyz_sparse.detach().clone()
                    v = init_v.detach().clone()
                    # Reset control points
                    ctrl_curr = init_ctrl.detach().clone()
                    ctrl_prev = init_ctrl.detach().clone()
                    simulator.init_controller(ctrl_curr)
                    # Reset step counter and render
                    step_id = 0
                    render_and_show()
                    just_reset = True
                    continue

                # Controller movement is handled in per-frame polling for multi-key input.
                if event.key in keymap:
                    pass

        # Poll keyboard: hold ESC to quit
        if running:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                running = False

        # Skip simulation step if just reset
        if not running or just_reset:
            continue

        # Per-frame polling: aggregate multi-key displacement for two control points
        keys = pygame.key.get_pressed()
        delta0 = np.zeros(3, dtype=np.float32)
        delta1 = np.zeros(3, dtype=np.float32)
        for k, (cid, delta) in keymap.items():
            if keys[k]:
                if cid == 0:
                    delta0 += delta
                elif cid == 1:
                    delta1 += delta
        if (delta0 != 0).any() or (delta1 != 0).any():
            ctrl_prev = ctrl_curr.detach().clone()
            if (delta0 != 0).any():
                ctrl_curr[0] = ctrl_curr[0] + torch.tensor(delta0, device=ctrl_curr.device, dtype=ctrl_curr.dtype)
            if (delta1 != 0).any():
                ctrl_curr[1] = ctrl_curr[1] + torch.tensor(delta1, device=ctrl_curr.device, dtype=ctrl_curr.dtype)
            moved_this_frame = True
        else:
            moved_this_frame = False

        # Set controller strategy based on movement, then run one simulation step
        if moved_this_frame:
            simulator.set_controller_target(ctrl_prev, ctrl_curr)
        else:
            # simulator.disable_controller_once()
            # Keep control points fixed and keep applying force.
            pass

        # One simulation step with NaN retry protection
        is_nan = True
        while is_nan:
            with torch.no_grad():
                xyz_all_o, xyz_o, v_o, is_nan = simulator(gaussians._xyz, xyz, v, step_id + 1)
            if is_nan:
                print("NaN detected, increasing internal step count and retrying...")
                simulator.n_step = simulator.n_step + cfg.DYNAMIC.STEP_ADD
                assert simulator.n_step <= cfg.DYNAMIC.MAX_N_STEP, "Dynamic simulation failed: exceeded internal-step limit"
            else:
                gaussians._xyz = xyz_all_o
                xyz = xyz_o.detach().clone()
                v = v_o.detach().clone()
                render_and_show()
                step_id += 1
                print('Step:', step_id)
        
        # Idle mode without simulation is disabled; now stepping every frame.
    
    pygame.quit()
    print("Exit.")
    if save_dir is not None and len(video_frames) > 0:
        imageio.mimwrite(str(save_dir / 'demo.mp4'), video_frames[:1000], fps=30, quality=8)


if __name__ == '__main__':
    main()

