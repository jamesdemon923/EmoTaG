"""Paper-protocol rendering: the audio-driven head composited over the per-frame
torso+background, evaluated against the original full frames.

This matches the standard talking-head evaluation (ER-NeRF / InsTaG / EmoTaG
paper): the model synthesizes the head; the torso and background come from the
source frames, and PSNR/SSIM/LPIPS are computed on full frames.
"""
import copy
import os
import sys
from argparse import ArgumentParser

import imageio
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render_motion
from scene import GaussianModel, MotionNetwork, Scene
from scene.flame_wrapper import SimpleFlameWrapper


def _person_matte(source_path, img_id, H):
    """Per-frame person matte [1,H,W] in 0..1 from the scene's matting alphas
    (data/<ID>/matte/<img_id>.png, the tracker's foreground masks). Used by
    bg_mode gt to trim the silhouette: the avatar was trained against
    white-matted frames through a dilated head mask, so its outermost silhouette
    gaussians carry a white rim that only shows over a real background."""
    p = os.path.join(source_path, "matte", f"{img_id}.png")
    if not os.path.exists(p):
        return None
    from scipy.ndimage import gaussian_filter, grey_erosion
    m = imageio.imread(p).astype(np.float32) / 255.0
    if m.ndim == 3:
        m = m[..., 0]
    # Harden, erode a little, then feather: the matting alpha decays softly over the
    # rim zone (~30% opacity), which would blend the white rim into the background
    # as a milky halo. Binarizing at a high threshold and pulling the edge inward
    # replaces the rim completely; the hair loses <=2px of soft wisps.
    m = (m > 0.75).astype(np.float32)
    m = grey_erosion(m, size=5)
    m = gaussian_filter(m, sigma=1.2)
    t = torch.from_numpy(m).float().unsqueeze(0)
    if m.shape[0] != H:
        t = torch.nn.functional.interpolate(t.unsqueeze(0), size=(H, H), mode="bilinear")[0]
    return t.clamp(0, 1)


def _neck_row(face_mask, H):
    """Row of the neck = narrowest person row in the anatomical band below the head.
    Below this row is torso+background (taken from GT); above is the synthesized head."""
    if face_mask is None:
        return None
    import math
    from utils.mask_utils import unpack_bool_mask
    from scipy.ndimage import uniform_filter1d
    m = np.asarray(unpack_bool_mask(face_mask))
    s = int(math.isqrt(m.size))
    if s * s != m.size:
        return None
    m = m.reshape(s, s)
    w = uniform_filter1d(m.sum(1).astype(float), 5)
    top = np.where(w > 3)[0]
    if len(top) == 0:
        return None
    t0 = top[0]
    hw = np.median(w[t0:t0 + 70])
    lo = int(t0 + 0.85 * hw)
    hi = min(int(t0 + 1.5 * hw), s - 1)
    if lo >= hi:
        return None
    nk = lo + int(np.argmin(w[lo:hi + 1]))
    return int(round(nk * H / s)) if s != H else nk


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--out_name", default="comp")
    parser.add_argument("--eval_from", type=int, default=0,
                        help="Write only frames >= this index (e.g. 125 = the held-out "
                             "test segment after a 5s/125-frame adaptation).")
    parser.add_argument("--bg_mode", choices=["plate", "gt"], default="plate",
                        help="plate (default): head over the scene's torso/background "
                             "plate with the below-neck region from the source frames "
                             "(the paper metric). gt: composite the head directly over "
                             "the full source frame, keeping whatever background it has.")
    parser.add_argument("--drive_with_gt", action="store_true")
    parser.add_argument("--smooth_sigma", type=float, default=1.0,
                        help="Temporal Gaussian sigma (frames) for smoothing predicted jaw and mouth deltas; 0 disables.")
    parser.add_argument("--smooth_sigma_exp", type=float, default=1.0,
                        help="Sigma for the 50-dim expression (upper face tolerates stronger smoothing than jaw).")
    parser.add_argument("--canonical_from", default=None,
                        help="Scene dir whose canonical mesh defines the binding's canonical "
                             "frames. For cross-clip driving this must be the ADAPTATION scene: "
                             "the avatar's local offsets were learned against ITS canonical "
                             "orientation, and two independent trackings of the same person "
                             "canonicalize slightly differently (visible as a rigid ghosting "
                             "offset under extreme expressions).")
    args = get_combined_args(parser)
    dataset, pipe = model.extract(args), pipeline.extract(args)

    with torch.no_grad():
        dataset.type = "face"
        gaussians = GaussianModel(copy.deepcopy(dataset))
        dataset.model_center = np.load(os.path.join(dataset.source_path, "model_center.npy"))
        gaussians.flame_wrapper = SimpleFlameWrapper(
            flame_params_file=os.path.join(dataset.source_path, "flame_params.npz")).cuda()
        idp = os.path.join(dataset.source_path, "identity_feature.npy")
        if os.path.exists(idp):
            gaussians.identity_feature = torch.from_numpy(np.load(idp).reshape(-1)).float().cuda()
        scene = Scene(dataset, gaussians, shuffle=False)
        identity_index_path = os.path.join(dataset.model_path, "identity_index.txt")
        if os.path.exists(identity_index_path):
            with open(identity_index_path) as f:
                gaussians.identity_index = int(f.read().strip())
        motion_net = MotionNetwork(args=dataset).cuda()
        model_params, motion_params, _, _ = torch.load(
            os.path.join(dataset.model_path, "chkpnt_face_latest.pth"))
        motion_net.load_state_dict(motion_params, strict=False)
        gaussians.restore(model_params, None)
        # Expression-conditioned deformation prior: load the shared decoder and this
        # identity's fitted code if the adaptation produced one.
        deform_code_path = os.path.join(dataset.model_path, "deform_code.pth")
        deform_dec_path = os.path.join(dataset.model_path, "deform_decoder.pth")
        if os.path.exists(deform_code_path) and os.path.exists(deform_dec_path):
            from scene.deform_decoder import ExprDeformDecoder
            dd = ExprDeformDecoder().cuda()
            dd.load_state_dict(torch.load(deform_dec_path, map_location="cuda:0"))
            dd.eval()
            gaussians.deform_decoder = dd
            gaussians.deform_code = torch.load(deform_code_path, map_location="cuda:0").cuda()
            print("loaded expression deformation prior + identity code")
        if getattr(args, 'canonical_from', None):
            can = np.load(os.path.join(args.canonical_from, "vertices.npy"))
            b = gaussians.flame_binding
            b.canonical_mesh_vertices = torch.tensor(can).float().to(b.canonical_mesh_vertices.device)
            b._canonical_frames = None
            b._update_canonical_points()
            print(f"canonical mesh overridden from {args.canonical_from}")
        # Render over BLACK so pkg['render'] is premultiplied foreground RGB and
        # compositing is rgb + (1-alpha)*plate.
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        views = scene.getTestCameras()
        # Pass 1: collect per-frame predicted exp/jaw, then smooth over time with a
        # small Gaussian kernel. The center-frame prediction (which fixed the lip-sync
        # low-pass) carries frame-to-frame parameter tremor (~30% above the tracked
        # GT's temporal variation) that reads as facial jitter; sigma~1 frame removes
        # the tremor while leaving articulation timing intact.
        smoothed = None
        sigma = float(getattr(args, "smooth_sigma", 1.0) or 0)
        if sigma > 0 and not args.drive_with_gt:
            exps, jaws = [], []
            for view in views:
                auds = view.talking_dict['auds'].cuda()
                au = view.talking_dict['au_features'].cuda()
                ident = getattr(gaussians, 'identity_feature', None)
                code = None
                if getattr(motion_net, 'individual_dim', 0) > 0:
                    idx = int(getattr(gaussians, 'identity_index', 0))
                    code = motion_net.individual_codes[idx:idx + 1]
                preds = motion_net(auds, c=code, au=au, identity=ident)
                pe, pj = preds['flame_exp'], preds['flame_jaw']
                if pe.dim() > 1:
                    pe = pe[pe.shape[0] // 2]
                if pj.dim() > 1:
                    pj = pj[pj.shape[0] // 2]
                exps.append(pe.float().cpu().numpy())
                jaws.append(pj.float().cpu().numpy())
            from scipy.ndimage import gaussian_filter1d
            sigma_exp = float(getattr(args, 'smooth_sigma_exp', 0) or sigma)
            exps = gaussian_filter1d(np.stack(exps), sigma=sigma_exp, axis=0)
            jaws = gaussian_filter1d(np.stack(jaws), sigma=sigma, axis=0)
            smoothed = (torch.tensor(exps).float().cuda(), torch.tensor(jaws).float().cuda())
        # Pass 2: collect the intra-oral triplane deltas under the smoothed drive and
        # smooth them too (they are predicted per frame from audio and carry the same
        # frame-to-frame tremor as the expression head).
        mouth_seq = None
        if smoothed is not None:
            from scipy.ndimage import gaussian_filter1d
            collected = []
            for vi, view in enumerate(views):
                mp = render_motion(view, gaussians, motion_net, pipe, background,
                                   drive_params=(smoothed[0][vi], smoothed[1][vi]),
                                   return_motion_only=True)['mouth_pred']
                collected.append(mp)
            if collected and collected[0] is not None:
                keys = ['d_mu', 'd_rot', 'd_scale']
                stacks = {k: np.stack([c[k].float().cpu().numpy() for c in collected]) for k in keys}
                mouth_seq = {k: torch.tensor(gaussian_filter1d(v, sigma=sigma, axis=0)).float().cuda()
                             for k, v in stacks.items()}
        # bg_mode gt prefers data/<ID>/bg_imgs/ — the original (pre-matting) frames with
        # the real background, frame-aligned with the scene. Falls back to the matted
        # GT frame when absent.
        bg_dir = os.path.join(dataset.source_path, "bg_imgs")
        use_bg_imgs = args.bg_mode == "gt" and os.path.isdir(bg_dir)
        rframes, gframes = [], []
        for vi, view in enumerate(views):
            dp = (smoothed[0][vi], smoothed[1][vi]) if smoothed is not None else None
            mp = {k: v[vi] for k, v in mouth_seq.items()} if mouth_seq is not None else None
            pkg = render_motion(view, gaussians, motion_net, pipe, background,
                                drive_with_gt=args.drive_with_gt, drive_params=dp, mouth_params=mp)
            rgb = pkg["render"].clamp(0, 1)
            alpha = pkg["alpha"].clamp(0, 1)
            plate = view.background.cuda().float() / 255.0  # torso composited over bc.jpg
            gt = view.original_image.cuda().float()
            if gt.max() > 1.5:
                gt = gt / 255.0
            gt = gt.clamp(0, 1)
            # bg_mode gt: the source frame itself is the backdrop, so the render keeps
            # whatever background the video has. Default: the scene's torso/bc plate.
            if use_bg_imgs:
                bg_path = os.path.join(bg_dir, f"{view.talking_dict['img_id']}.jpg")
                if os.path.exists(bg_path):
                    gt = torch.from_numpy(imageio.imread(bg_path).astype(np.float32) / 255.0
                                          ).permute(2, 0, 1).cuda().clamp(0, 1)
            backdrop = gt if args.bg_mode == "gt" else plate
            comp = rgb + backdrop * (1.0 - alpha)
            # Standard protocol: the model synthesizes the head; torso+background
            # below the neck come from the source frames (EMOTAG_GT_TORSO=0 disables,
            # for inspecting the raw render).
            if os.environ.get('EMOTAG_GT_TORSO', '1') != '0':
                nk = _neck_row(view.talking_dict.get('face_mask'), comp.shape[-2])
                if nk is not None:
                    # Feathered blend across the neck (not a hard cut) so the synthesized
                    # head fades smoothly into the GT torso with no visible seam.
                    Hc = comp.shape[-2]
                    band = max(8, int(0.05 * Hc))          # ~5% of height transition
                    rows = torch.arange(Hc, device=comp.device, dtype=comp.dtype)
                    # weight = 1 (keep head) above nk-band, 0 (use GT) below nk+band, smooth between
                    t = ((rows - (nk - band)) / (2 * band)).clamp(0, 1)
                    wgt = (1 - (3 * t ** 2 - 2 * t ** 3)).view(1, Hc, 1)   # smoothstep, 1->0
                    comp = comp * wgt + gt * (1 - wgt)
            # Trim by the per-frame person matte (when matte/ exists): replaces any
            # residual silhouette band with the backdrop.
            pm = _person_matte(dataset.source_path, view.talking_dict['img_id'], comp.shape[-2])
            if pm is not None:
                pm = pm.to(comp.device)
                outside = gt if args.bg_mode == "gt" else plate
                comp = comp * pm + outside * (1 - pm)
            rframes.append((comp.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            gframes.append((gt.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        start = max(0, int(getattr(args, "eval_from", 0) or 0))
        if start:
            rframes, gframes = rframes[start:], gframes[start:]
        out = os.path.join(dataset.model_path, "test")
        os.makedirs(out, exist_ok=True)
        imageio.mimwrite(os.path.join(out, f"rendered_{args.out_name}.mp4"), rframes, fps=25, quality=8, macro_block_size=1)
        imageio.mimwrite(os.path.join(out, f"gt_{args.out_name}.mp4"), gframes, fps=25, quality=8, macro_block_size=1)
        print("saved", len(rframes), "frames to", out)


if __name__ == "__main__":
    main()
