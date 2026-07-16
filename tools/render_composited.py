"""Paper-protocol rendering: audio-driven synthesis composited over the scene's
static torso+background plate, evaluated against the ORIGINAL full frames.

This matches the standard talking-head evaluation (ER-NeRF / InsTaG / EmoTaG
paper): rendered head alpha-composited onto the per-frame torso+background,
PSNR/SSIM/LPIPS computed on full frames. The green-screen comparison used by
synthesize.py is a stricter internal debugging view, not the paper metric.
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


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--out_name", default="comp")
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
        if getattr(args, 'canonical_from', None):
            can = np.load(os.path.join(args.canonical_from, "vertices.npy"))
            b = gaussians.flame_binding
            b.canonical_mesh_vertices = torch.tensor(can).float().to(b.canonical_mesh_vertices.device)
            b._canonical_frames = None
            b._update_canonical_points()
            print(f"canonical mesh overridden from {args.canonical_from}")
        # Render over BLACK so pkg['render'] is premultiplied foreground RGB;
        # compositing is then rgb + (1-alpha)*plate with no background-colour
        # pollution in the semi-transparent silhouette fringe.
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
        rframes, gframes = [], []
        for vi, view in enumerate(views):
            dp = (smoothed[0][vi], smoothed[1][vi]) if smoothed is not None else None
            mp = {k: v[vi] for k, v in mouth_seq.items()} if mouth_seq is not None else None
            pkg = render_motion(view, gaussians, motion_net, pipe, background,
                                drive_with_gt=args.drive_with_gt, drive_params=dp, mouth_params=mp)
            rgb = pkg["render"].clamp(0, 1)
            alpha = pkg["alpha"].clamp(0, 1)
            plate = view.background.cuda().float() / 255.0  # torso composited over bc.jpg
            comp = rgb + plate * (1.0 - alpha)
            rframes.append((comp.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            gt = view.original_image.cuda().float()
            if gt.max() > 1.5:
                gt = gt / 255.0
            gframes.append((gt.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        out = os.path.join(dataset.model_path, "test")
        os.makedirs(out, exist_ok=True)
        imageio.mimwrite(os.path.join(out, f"rendered_{args.out_name}.mp4"), rframes, fps=25, quality=8, macro_block_size=1)
        imageio.mimwrite(os.path.join(out, f"gt_{args.out_name}.mp4"), gframes, fps=25, quality=8, macro_block_size=1)
        print("saved", len(rframes), "frames to", out)


if __name__ == "__main__":
    main()
