# calculate psnr, ssim, lpips

import os
import cv2
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import imageio

def read_images_from_dir(directory):
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            images.append(img)
    return images

def read_images_from_video(video_path):
    images = []
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    while success:
        images.append(image)
        success, image = vidcap.read()
    vidcap.release()
    return images
def crop_img_by_column(img, num_col=1, column_list=[0]):
    height, width, _ = img.shape
    single_img_width = width // num_col
    cropped_imgs = []
    for i in column_list:
        x1 = single_img_width * i
        x2 = single_img_width * (i + 1)
        cropped_img = img[:, x1:x2]
        cropped_imgs.append(cropped_img)
    return cropped_imgs

def cal_one_video_psnr_ssim_lpips(gt_images, gen_images):
    psnr_score = 0
    ssim_score = 0
    lpips_score = 0
    loss_fn_alex = lpips.LPIPS(net='alex')
    
    for gt_image, gen_image in zip(gt_images, gen_images):
        # 
        psnr_score += psnr(gt_image, gen_image)
        
        # 
        ssim_score += ssim(gt_image, gen_image, multichannel=True,channel_axis=2)
        
        # 
        gt_image_tensor = torch.tensor(gt_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        gen_image_tensor = torch.tensor(gen_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        lpips_score += loss_fn_alex(gt_image_tensor, gen_image_tensor).item()
    
    num_images = len(gt_images)
    psnr_score /= num_images
    ssim_score /= num_images
    lpips_score /= num_images

    return psnr_score, ssim_score, lpips_score

from typing import Tuple
import scipy
from scipy.optimize import linear_sum_assignment
"""translated: https://github.com/universome/fvd-comparison"""
def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0) # [d]
    sigma = np.cov(feats, rowvar=False) # [d, d]
    return mu, sigma

def compute_fvd(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    return float(fid)

@torch.no_grad()
def compute_our_fvd(videos_fake: np.ndarray, videos_real: np.ndarray, device: str='cuda') -> float:
    # detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
    detector_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.
    
    # from util import open_url
    # with open_url(detector_url, verbose=False) as f:
    #     detector = torch.jit.load(f).eval().to(device)
        
    detector_file_path = os.environ.get('RECONPHYS_I3D_PATH', '')
    if not detector_file_path or not os.path.isfile(detector_file_path):
        raise FileNotFoundError(
            "FVD detector not found. Set RECONPHYS_I3D_PATH to an i3d_torchscript.pt path."
        )
    detector = torch.jit.load(detector_file_path).eval().to(device)
    videos_fake = torch.from_numpy(videos_fake).permute(0, 4, 1, 2, 3).to(device)
    videos_real = torch.from_numpy(videos_real).permute(0, 4, 1, 2, 3).to(device)

    feats_fake = detector(videos_fake, **detector_kwargs).cpu().numpy()

    feats_real = detector(videos_real, **detector_kwargs).cpu().numpy()
    # [2,400,8,8]->[2,400]
    feats_fake = np.mean(feats_fake, axis=(2, 3))
    feats_real = np.mean(feats_real, axis=(2, 3))
    # print(feats_fake.shape)
    return compute_fvd(feats_fake, feats_real)


class VideoEvaluator:
    """
    translated LPIPS/I3D translated：
    - cal_one_video_psnr_ssim_lpips: translated
    - compute_our_fvd: translated I3D translated FVD
    """
    def __init__(
        self,
        lpips_net: str = 'alex',
        lpips_device: str = None,
        fvd_device: str = None,
        detector_file_path: str = ''
    ):
        self.lpips_device = lpips_device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.fvd_device = fvd_device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 
        self.lpips_fn = lpips.LPIPS(net=lpips_net).to(self.lpips_device).eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad_(False)

        # 
        if not detector_file_path:
            detector_file_path = os.environ.get('RECONPHYS_I3D_PATH', '')
        self.detector_file_path = detector_file_path
        self.detector = None
        self.detector_kwargs = dict(rescale=False, resize=False, return_features=True)

    def _get_detector(self):
        if self.detector is None:
            if not self.detector_file_path or not os.path.isfile(self.detector_file_path):
                raise FileNotFoundError(
                    "FVD detector not found. Set RECONPHYS_I3D_PATH or pass detector_file_path explicitly."
                )
            self.detector = torch.jit.load(self.detector_file_path).eval().to(self.fvd_device)
            for p in self.detector.parameters():
                p.requires_grad_(False)
        return self.detector

    @torch.no_grad()
    def cal_one_video_psnr_ssim_lpips(self, gt_images, gen_images):
        """
        Args:
            gt_images, gen_images: List[np.ndarray(H,W,3)], uint8/float translated。translated，translated。
        Returns:
            (psnr_avg, ssim_avg, lpips_avg)
        """
        psnr_sum, ssim_sum, lpips_sum = 0.0, 0.0, 0.0
        cnt = 0

        for gt_image, gen_image in zip(gt_images, gen_images):
            # PSNR
            psnr_sum += psnr(gt_image, gen_image)

            # 
            ssim_sum += ssim(gt_image, gen_image, channel_axis=2)

            # 
            gt_tensor = torch.tensor(gt_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            gen_tensor = torch.tensor(gen_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            gt_tensor = gt_tensor.to(self.lpips_device)
            gen_tensor = gen_tensor.to(self.lpips_device)
            lpips_sum += self.lpips_fn(gt_tensor, gen_tensor).item()

            cnt += 1

        cnt = max(cnt, 1)
        return psnr_sum / cnt, ssim_sum / cnt, lpips_sum / cnt

    @torch.no_grad()
    def compute_our_fvd(self, videos_fake: np.ndarray, videos_real: np.ndarray) -> float:
        """
        Args:
            videos_fake, videos_real: np.ndarray [N, T, H, W, 3] translated [0,1] translated
        Returns:
            fvd: float
        """
        detector = self._get_detector()

        vf = torch.from_numpy(videos_fake).permute(0, 4, 1, 2, 3).to(self.fvd_device)  # [N,3,T,H,W]
        vr = torch.from_numpy(videos_real).permute(0, 4, 1, 2, 3).to(self.fvd_device)

        feats_fake = detector(vf, **self.detector_kwargs).cpu().numpy()
        feats_real = detector(vr, **self.detector_kwargs).cpu().numpy()

        # [N, C, Hf, Wf] -> [N, C]
        feats_fake = np.mean(feats_fake, axis=(2, 3))
        feats_real = np.mean(feats_real, axis=(2, 3))
        return compute_fvd(feats_fake, feats_real)

# Chamfer Distance
def subsample_points(pts: torch.Tensor, max_points: int) -> torch.Tensor:
    """
    pts: [N,3] -> [M,3], M<=max_points
    """
    if pts.ndim != 2 or pts.shape[-1] != 3:
        raise ValueError(f"points shape should be [N,3], got {tuple(pts.shape)}")
    N = pts.shape[0]
    if N <= max_points:
        return pts
    idx = torch.randperm(N, device=pts.device)[:max_points]
    return pts.index_select(0, idx)


def chamfer_distance(x: torch.Tensor, y: torch.Tensor, squared: bool = True) -> torch.Tensor:
    """
    x: [Nx,3], y: [Ny,3] -> translated Chamfer
    """
    d = torch.cdist(x, y, p=2)
    if squared:
        d = d.pow(2)
    cd_xy = d.min(dim=1).values.mean()
    cd_yx = d.min(dim=0).values.mean()
    return 0.5 * (cd_xy + cd_yx)

@torch.no_grad()
def earth_movers_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    squared: bool = True,
) -> torch.Tensor:
    """
    translated EMD（translated）。
    - translated x,y translated [N,3]/[M,3]，translated N==M。
    - translated N!=M，translated。
    - translated：Hungarian translated O(N^3)，translated N<=1024（translated<=512）。

    translated：translated torch.Tensor（translated x translated/translated）
    """
    if x.ndim != 2 or y.ndim != 2 or x.shape[-1] != 3 or y.shape[-1] != 3:
        raise ValueError(f"EMD inputs must be [N,3] and [M,3], got {tuple(x.shape)} and {tuple(y.shape)}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"EMD requires equal number of points, got {x.shape[0]} and {y.shape[0]}")

    # 
    # 
    cost = torch.cdist(x, y, p=2)  # [N,N]
    if squared:
        cost = cost.pow(2)
    cost_np = cost.detach().cpu().numpy()

    row_ind, col_ind = linear_sum_assignment(cost_np)  # 
    emd = float(cost_np[row_ind, col_ind].mean())

    return torch.as_tensor(emd, dtype=x.dtype, device=x.device)
