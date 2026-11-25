"""Render an adapted EmoTaG avatar as a full-frame talking-head video.

The head is synthesized; the torso and the background come from the source frames.
Outputs in <model>/test/: rendered_<name>.mp4, gt_<name>.mp4, the same with audio,
masks_<name>.npz (head region) and motion_<name>.npz (predicted FLAME motion).

    python tools/render_composited.py -s data/<ID> -m output/<ID> --eval_from 125
    # novel audio
    python tools/render_composited.py -s data/<ID> -m output/<ID> --audio_file new_whisper.npy --wav new.wav --N_views -1
"""
import copy
import glob
import os
import subprocess
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
from utils.drive_sanitize import (TRACKED_CONVENTION, drive_convention, gaze_range, has_refit_eyelids,
                                  load_drive_params, raw_drive)

DEFAULT_GAIN_SINGLE = "1.4,1.2"
DEFAULT_GAIN_ENSEMBLE = "1.7,1.5"
EYELID_MARGIN = 0.1


def _adaptation_clips(model_path, adapt_src):
    """[(clip directory, adaptation frame ids)] recorded by adapt_emotag.py in the model directory."""
    import json
    setup = os.path.join(model_path, "adaptation_setup.json")
    if os.path.exists(setup):
        def _find(path):
            if not os.path.isdir(path) and adapt_src:
                cand = os.path.join(os.path.dirname(os.path.abspath(adapt_src)), os.path.basename(str(path).rstrip("/")))
                if os.path.isdir(cand):
                    return cand
            return path
        return [(_find(c["source_path"]), c["frame_ids"]) for c in json.load(open(setup))["clips"]]
    if adapt_src and os.path.exists(os.path.join(model_path, "adaptation_frames.txt")):
        return [(adapt_src, [int(x) for x in open(os.path.join(model_path, "adaptation_frames.txt")).read().split()])]
    return []


def check_drive_convention(model_path, adapt_src, source_path):
    """Compare the FLAME parameter convention the avatar was adapted on (adaptation_setup.json."""
    import json
    setup = os.path.join(model_path, "adaptation_setup.json")
    model_conv = TRACKED_CONVENTION
    if os.path.exists(setup):
        model_conv = json.load(open(setup)).get("drive_convention") or TRACKED_CONVENTION
    drive_conv = drive_convention(os.path.join(source_path, "flame_params.npz")) or TRACKED_CONVENTION
    if drive_conv != model_conv:
        print(f"[render] WARNING: FLAME parameter convention mismatch: the avatar was adapted on '{model_conv}' "
              f"parameters, the driving clip {source_path} holds '{drive_conv}' parameters. Mouth and eyelid "
              f"shapes mean different things in the two; re-adapt the avatar or use a driving clip of the same "
              f"convention (tools/refine_landmarks.py).")
    for src, _ in _adaptation_clips(model_path, adapt_src):
        now = drive_convention(os.path.join(src, "flame_params.npz"))
        if now is not None and now != model_conv:
            print(f"[render] WARNING: the adaptation clip {src} now holds '{now}' FLAME parameters, but the avatar "
                  f"was adapted on '{model_conv}': statistics read from that clip (mouth-motion distribution, gaze "
                  f"and eyelid range) no longer describe the avatar. Re-adapt it.")
    return model_conv, drive_conv


def adaptation_gaze_range(model_path, adapt_src):
    """eyes_pose range covered by the adaptation frames, used to clamp the driving gaze exactly as adapt_emotag.py."""
    import json
    setup = os.path.join(model_path, "adaptation_setup.json")
    if os.path.exists(setup):
        stored = json.load(open(setup)).get("gaze_range")
        if stored:
            return np.asarray(stored[0], np.float32), np.asarray(stored[1], np.float32)
    clips = _adaptation_clips(model_path, adapt_src)
    if not clips or not all(os.path.exists(os.path.join(c, "flame_params.npz")) for c, _ in clips):
        return None
    return gaze_range([(load_drive_params(os.path.join(c, "flame_params.npz"), isolate=ids)[0], ids) for c, ids in clips])


def identity_motion_stats(model_path, adapt_src, upper_proj):
    """Distribution of this identity's tracked mouth motion over its adaptation frames."""
    import hashlib
    import json
    cache = os.path.join(model_path, "motion_stats.json")
    clips = _adaptation_clips(model_path, adapt_src)
    fingerprint = [[os.path.basename(os.path.normpath(c)), [int(i) for i in ids]] for c, ids in clips]
    conventions = [drive_convention(os.path.join(c, "flame_params.npz")) for c, _ in clips]
    if any(cv not in (None, TRACKED_CONVENTION) for cv in conventions):
        fingerprint.append(conventions)
    key = hashlib.sha1(json.dumps(fingerprint).encode()).hexdigest() if clips else None
    reachable = bool(clips) and all(os.path.exists(os.path.join(c, "flame_params.npz")) for c, _ in clips)
    if os.path.exists(cache):
        try:
            st = json.load(open(cache))
            if not reachable or key is None or st.get("key") == key:
                return st
            print("[render] motion_stats.json has no or another adaptation fingerprint: recomputing")
        except ValueError:
            pass
    E, J = [], []
    for src, ids in clips:
        fpz = os.path.join(src, "flame_params.npz")
        if not os.path.exists(fpz):
            return None
        fp = load_drive_params(fpz, isolate=ids)[0]
        for i in ids:
            if i in fp:
                d = fp[i]
                E.append(np.asarray(d["expression"]).reshape(-1)[:upper_proj.shape[0]])
                J.append(np.asarray(d["jaw_pose"]).reshape(-1)[:3])
    if len(J) < 25:
        return None
    from utils.flame_regions import expression_region_bases
    E, J = np.stack(E), np.stack(J)
    low = E @ (np.eye(E.shape[1]) - upper_proj).T
    bl = expression_region_bases(E.shape[1])[0]
    st = {"n": int(len(J)), "jaw_open_p10": float(np.percentile(J[:, 0], 10)),
          "jaw_open_p90": float(np.percentile(J[:, 0], 90)), "jaw_open_min": float(J[:, 0].min()),
          "lower_mean": low.mean(0).tolist(),
          "lower_std": float(np.sqrt((((low - low.mean(0)) @ bl.T) ** 2).mean())), "key": key}
    try:
        tmp = f"{cache}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(st, f)
        os.replace(tmp, cache)
    except OSError:
        pass
    return st


def match_identity_motion(exps, jaws, st, upper_proj, max_ratio=3.0):
    """Map the audio-driven motion onto the identity's own distribution."""
    from utils.flame_regions import expression_region_bases
    jaws, exps = jaws.copy(), exps.copy()
    j0 = jaws[:, 0]
    q10, q90 = np.percentile(j0, [10, 90])
    slope = float(np.clip((st["jaw_open_p90"] - st["jaw_open_p10"]) / max(q90 - q10, 1e-6), 1 / max_ratio, max_ratio))
    jaws[:, 0] = np.maximum(st["jaw_open_p10"] + (j0 - q10) * slope, st["jaw_open_min"])
    L = np.eye(exps.shape[1]) - upper_proj
    low = exps @ L.T
    dev = low - low.mean(0, keepdims=True)
    bl = expression_region_bases(exps.shape[1])[0]
    sp = float(np.sqrt(((dev @ bl.T) ** 2).mean()))
    scale = float(np.clip(st["lower_std"] / max(sp, 1e-8), 1 / max_ratio, max_ratio))
    exps = exps - low + np.asarray(st["lower_mean"])[None] + scale * dev
    return exps, jaws, slope, scale


def _load_matte(source_path, img_id, H):
    p = os.path.join(source_path, "matte", f"{img_id}.png")
    if not os.path.exists(p):
        return None
    m = imageio.imread(p).astype(np.float32) / 255.0
    if m.ndim == 3:
        m = m[..., 0]
    t = torch.from_numpy(m).float().unsqueeze(0)
    if m.shape[0] != H:
        t = torch.nn.functional.interpolate(t.unsqueeze(0), size=(H, H), mode="bilinear")[0]
    return t.clamp(0, 1)


def _estimate_bg_plate(source_path, cache_path, step=5):
    """Static background plate (ER-NeRF style bc.jpg) from the original frames."""
    if cache_path and os.path.exists(cache_path):
        return imageio.imread(cache_path).astype(np.float32) / 255.0
    import cv2
    bg_dir, matte_dir = os.path.join(source_path, "bg_imgs"), os.path.join(source_path, "matte")
    ids = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(bg_dir) if f.endswith(".jpg"))[::step]
    imgs, valid = [], []
    for i in ids:
        mp = os.path.join(matte_dir, f"{i}.png")
        if not os.path.exists(mp):
            continue
        imgs.append(imageio.imread(os.path.join(bg_dir, f"{i}.jpg")).astype(np.float32))
        m = imageio.imread(mp).astype(np.float32) / 255.0
        if m.ndim == 3:
            m = m[..., 0]
        valid.append(cv2.dilate((m > 0.02).astype(np.uint8), np.ones((9, 9), np.uint8)) == 0)
    if len(imgs) < max(3, len(ids) // 2):
        raise FileNotFoundError(f"[render] too few matted frames under {matte_dir} to estimate a background plate; "
                                f"use --bg_mode white")
    imgs, valid = np.stack(imgs), np.stack(valid)
    stack = np.where(valid[..., None], imgs, np.inf)
    stack.sort(axis=0)
    n = np.broadcast_to(valid.sum(0)[..., None], stack.shape[1:])
    lo = np.take_along_axis(stack, np.maximum((n - 1) // 2, 0)[None], 0)[0]
    hi = np.take_along_axis(stack, np.maximum(n // 2, 0)[None], 0)[0]
    plate = np.where(n > 0, (lo + hi) / 2, np.nan).astype(np.float32)
    hole = np.isnan(plate[..., 0])
    plate = np.nan_to_num(plate).clip(0, 255).astype(np.uint8)
    if hole.any():
        plate = cv2.inpaint(plate, hole.astype(np.uint8), 7, cv2.INPAINT_TELEA)
    if cache_path:
        tmp = f"{cache_path}.{os.getpid()}.tmp.png"
        imageio.imwrite(tmp, plate)
        os.replace(tmp, cache_path)
    return plate.astype(np.float32) / 255.0


def _trim_matte(matte):
    """Cosmetic only (--trim_matte)."""
    from scipy.ndimage import gaussian_filter, grey_erosion
    m = (matte[0].cpu().numpy() > 0.75).astype(np.float32)
    m = gaussian_filter(grey_erosion(m, size=5), sigma=1.2)
    return torch.from_numpy(m).float().unsqueeze(0)


def _neck_row(face_mask, H):
    """Row of the neck = narrowest person row in the anatomical band below the head."""
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


def _neck_row_from_landmarks(source_path, img_id, H, margin=0.12):
    """Row below which the torso is taken from the source frame."""
    p = os.path.join(source_path, "ori_imgs", f"{img_id}.lms")
    if not os.path.exists(p):
        return None
    lms = np.loadtxt(p).reshape(-1, 2)
    if lms.shape[0] < 68:
        return None
    chin = lms[4:13, 1].max()
    brow = lms[17:27, 1].min()
    row = int(round(chin + margin * max(chin - brow, 1.0)))
    if row >= H - 1:
        return None
    return max(row, 0)


def _torso_curve(source_path, img_id, H, W, nk, side_lift=0.0):
    """Per-column row below which the source frame (torso) is used."""
    p = os.path.join(source_path, "ori_imgs", f"{img_id}.lms")
    if nk is None or not os.path.exists(p):
        return None
    lms = np.loadtxt(p).reshape(-1, 2)
    if lms.shape[0] < 68:
        return None
    chin, brow = lms[4:13, 1].max(), lms[17:27, 1].min()
    fw = max(lms[16, 0] - lms[0, 0], 1.0)
    side = min(chin - side_lift * max(chin - brow, 1.0), nk)
    xin_l, xin_r = lms[5, 0], lms[11, 0]
    xout_l, xout_r = lms[2, 0] - 0.05 * fw, lms[14, 0] + 0.05 * fw
    x = np.arange(W, dtype=np.float32)
    t = np.where(x < xin_l, (xin_l - x) / max(xin_l - xout_l, 1.0), np.where(x > xin_r, (x - xin_r) / max(xout_r - xin_r, 1.0), 0.0))
    t = np.clip(t, 0.0, 1.0)
    return (nk + (side - nk) * (3 * t ** 2 - 2 * t ** 3)).astype(np.float32)


def _head_region(matte, face_mask, nk, H, dilate=8, curve=None):
    """GT head region used for region metrics."""
    from scipy.ndimage import binary_dilation
    if matte is not None:
        m = matte[0].cpu().numpy() > 0.5
    else:
        from utils.mask_utils import unpack_bool_mask
        m = np.asarray(unpack_bool_mask(face_mask)).reshape(H, H)
    below = None
    if nk is not None:
        below = np.zeros_like(m)
        if curve is None:
            below[nk:] = True
        else:
            below = np.arange(m.shape[0])[:, None] >= curve[None, :m.shape[1]]
        m[below] = False
    m = binary_dilation(m, iterations=dilate)
    if below is not None:
        m[below] = False
    return m


def _mux(video, wav, out, start_frame=0, fps=25):
    """Mux the driving audio (aligned to the written frame range) into a copy of video."""
    try:
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", video, "-ss", f"{start_frame / fps:.3f}", "-i", wav,
               "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", out]
        subprocess.run(cmd, check=True)
        return True
    except Exception as exc:
        print(f"[render] could not mux audio into {out}: {exc}")
        return False


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--out_name", default="comp")
    parser.add_argument("--eval_from", type=int, default=0,
                        help="Write only frames >= this index (e.g. 125 = the held-out "
                             "test segment after a 5s/125-frame adaptation).")
    parser.add_argument("--eval_to", type=int, default=-1)
    parser.add_argument("--bg_mode", choices=["real", "white", "plate", "gt"], default="real",
                        help="real (default, alias gt): background added back -- head over the video's "
                             "static background plate, compared with the original frames. "
                             "white (alias plate): head over the scene plate vs. the white-matted gt_imgs.")
    parser.add_argument("--frame_background", action="store_true",
                        help="Diagnostic (--bg_mode real only): fill the backdrop with each frame's own pixels where "
                             "the person matte is exactly 0 (pure background), the static plate elsewhere. No ground-"
                             "truth person pixel is used; measures how much the static plate costs in full-frame "
                             "metrics (dynamic backgrounds). Not the default protocol.")
    parser.add_argument("--torso_boundary", choices=["curve", "row"], default="curve",
                        help="Where the source frame (torso) replaces the render: 'row' = below one neck row (earlier "
                             "behaviour); 'curve' = the neck row under the jaw, rising to the chin level beside the face, "
                             "so the shoulders next to the neck come from the source frame too.")
    parser.add_argument("--torso_side_lift", type=float, default=0.2,
                        help="--torso_boundary curve: raise the boundary beside the face by this fraction of the face height.")
    parser.add_argument("--drive_with_gt", action="store_true",
                        help="Drive with the tracked FLAME expression/jaw (upper bound / debugging).")
    parser.add_argument("--smooth_sigma", type=float, default=None,
                        help="Temporal Gaussian sigma (frames) for the predicted jaw and mouth deltas; 0 disables. "
                             "Default: 0.75 for avatars adapted with --motion_steps (their motion network is not "
                             "over-fitted to the adaptation frames and needs less smoothing), 1.0 otherwise.")
    parser.add_argument("--smooth_sigma_exp", type=float, default=None,
                        help="Sigma for the 50-dim expression (default: --smooth_sigma); 0 disables.")
    parser.add_argument("--raw_drive", action="store_true",
                        help="A/B switch: use the raw tracked drive (no repair of glitched frames, no gaze median "
                             "filter / clamp) and restart the driving clip at frame 0 when the audio is longer "
                             "(instead of playing it back and forth). Same as EMOTAG_RAW_DRIVE=1.")
    parser.add_argument("--articulation_gain", default=None,
                        help="'jaw,lower_face' gains on the audio-driven motion about its mean, used by --motion_norm gain "
                             "(and by auto when no identity statistics are available; passing this option under auto "
                             "selects gain mode). Default: the inverse of the out-of-sample amplitude of 5-s "
                             "adaptations measured on held-out parts of the adaptation clips -- "
                             f"{DEFAULT_GAIN_SINGLE} for one model adapted from the Whisper prior (other priors are "
                             f"not calibrated: pass this option), {DEFAULT_GAIN_ENSEMBLE} with a motion ensemble. "
                             "Released / --au_mode legacy checkpoints default to 1,1 (they have no upper/lower-face "
                             "split, so an explicit expression gain scales the whole expression). '1,1' disables.")
    parser.add_argument("--motion_norm", choices=["auto", "identity", "gain", "none"], default="auto",
                        help="Post-processing of the audio-driven mouth motion. identity: map it onto the identity's "
                             "own tracked distribution over the adaptation frames (jaw-opening p10/p90, lower-face "
                             "mean and spread; robust to unfamiliar audio; AU-decoupled models only). gain: "
                             "--articulation_gain about the sequence mean. none: neither. auto (default): identity for "
                             "AU-decoupled models when the adaptation clip's flame_params.npz (or "
                             "<model>/motion_stats.json) is available, else gain; an explicit --articulation_gain "
                             "selects gain.")
    parser.add_argument("--drive_file", default=None,
                        help="Diagnostics: npz with 'exp' [N, 50] and 'jaw' [N, 3] (one row per frame of -s) used as the "
                             "FLAME expression / jaw drive instead of the audio-driven prediction.")
    parser.add_argument("--eyelids", choices=["pred", "frame"], default=None,
                        help="Eyelid state under audio drive. pred: as predicted by the motion network (from the "
                             "driving clip's action units). frame: copied from the driving frame's FLAME expression, "
                             "the way its head pose and gaze are (the eyelid coordinate of utils/flame_eyelids.py), "
                             "clamped to the range the adaptation frames covered. Default: frame when the driving "
                             "clip's eyelids were re-fitted to the image (tools/refine_landmarks.py; the tracker "
                             "does not close the lids), else pred.")
    parser.add_argument("--no_eyelid_clamp", action="store_true",
                        help="Diagnostics (--eyelids frame): do not clamp the copied eyelid state to the range covered "
                             "by the adaptation frames.")
    parser.add_argument("--eyelid_open_margin", type=float, default=0.0,
                        help="--eyelids frame: how far the copied eyelid state may go BELOW the adaptation range "
                             "(a wider-open eye than any adaptation frame), in units of the closure coordinate. "
                             "0 = never open the lids wider than the adaptation clip showed (a wider stare was "
                             "never observed and pulls lid gaussians over the iris); the closing side keeps "
                             "its margin.")
    parser.add_argument("--no_aux_motion", action="store_true",
                        help="Ignore <model>/aux_motion_*.pth (motion ensemble, tools/fit_aux_motion.py).")
    parser.add_argument("--aux_audio_file", default=None,
                        help="With --audio_file (novel audio) and a motion ensemble: features of the SAME wav for the "
                             "auxiliary encoder, e.g. new_w2v.npy from tools/extract_wav2vec2_features.py (default: "
                             "<audio_file stem>_<encoder>.npy, or aud_<encoder>.npy next to an aud_<encoder>.npy "
                             "--audio_file). Required when neither exists; or pass --no_aux_motion.")
    parser.add_argument("--keep_driving_identity", action="store_true",
                        help="Use the driving clip's canonical mesh / FLAME shape instead of the "
                             "avatar's own (only useful for debugging).")
    parser.add_argument("--identity_shape", action="store_true",
                        help="Also replace the driving clip's FLAME shape by the avatar's (same tracking scale only).")
    parser.add_argument("--canonical_from", default=None,
                        help="Override the canonical mesh with <dir>/vertices.npy (older checkpoints "
                             "that do not store the adaptation identity).")
    parser.add_argument("--trim_matte", action="store_true",
                        help="Cosmetic: erode the GT person matte over the composite (never for metrics).")
    parser.add_argument("--wav", default=None, help="Audio to mux into the *_audio.mp4 files "
                        "(default: <source>/aud.wav, or the wav next to --audio_file).")
    args = get_combined_args(parser)
    for key in ("canonical_from", "wav", "smooth_sigma", "smooth_sigma_exp", "articulation_gain", "aux_audio_file", "eyelids"):
        if not hasattr(args, key):
            setattr(args, key, None)
    dataset, pipe = model.extract(args), pipeline.extract(args)
    args.bg_mode = {"plate": "white", "real": "gt"}.get(args.bg_mode, args.bg_mode)
    if args.raw_drive:
        os.environ["EMOTAG_RAW_DRIVE"] = "1"

    import time as _time
    _T = [("start", _time.perf_counter())]
    with torch.no_grad():
        dataset.type = "face"
        gaussians = GaussianModel(copy.deepcopy(dataset))
        dataset.model_center = np.load(os.path.join(dataset.source_path, "model_center.npy"))
        model_params, motion_params, _, _ = torch.load(
            os.path.join(dataset.model_path, "chkpnt_face_latest.pth"), weights_only=False)
        gaussians.flame_wrapper = SimpleFlameWrapper(
            flame_params_file=os.path.join(dataset.source_path, "flame_params.npz"),
            add_teeth=GaussianModel.checkpoint_intraoral_mode(model_params) == "rigid").cuda()
        idp = os.path.join(dataset.source_path, "identity_feature.npy")
        adapt_src = None
        cfg = os.path.join(dataset.model_path, "cfg_args")
        if os.path.exists(cfg):
            try:
                from argparse import Namespace
                adapt_src = eval(open(cfg).read()).source_path
            except Exception:
                adapt_src = None
        if adapt_src and not os.path.isdir(adapt_src):
            _cand = os.path.join(os.path.dirname(os.path.abspath(dataset.source_path)), os.path.basename(str(adapt_src).rstrip("/")))
            if os.path.isdir(_cand):
                adapt_src = _cand
        if adapt_src and os.path.exists(os.path.join(adapt_src, "identity_feature.npy")):
            idp = os.path.join(adapt_src, "identity_feature.npy")
        elif adapt_src and os.path.abspath(adapt_src) != os.path.abspath(dataset.source_path):
            print(f"[render] WARNING: adaptation scene {adapt_src} (from cfg_args) not found; using the "
                  f"driving scene's identity_feature.npy. Pass the adaptation scene's path in cfg_args.")
        if os.path.exists(idp):
            gaussians.identity_feature = torch.from_numpy(np.load(idp).reshape(-1)).float().cuda()
        _, drive_conv = check_drive_convention(dataset.model_path, adapt_src, dataset.source_path)
        if args.eyelids is None:
            args.eyelids = "frame" if has_refit_eyelids(drive_conv) else "pred"
        if not raw_drive():
            for _c, _ids in _adaptation_clips(dataset.model_path, adapt_src):
                if os.path.realpath(_c) == os.path.realpath(dataset.source_path):
                    gaussians.flame_wrapper.isolate_frames(_ids)
            rng = adaptation_gaze_range(dataset.model_path, adapt_src)
            if rng is not None:
                gaussians.flame_wrapper.clamp_gaze(*rng)
            else:
                print("[render] WARNING: adaptation frames unknown (no adaptation_setup.json / tracking of the "
                      "adaptation clip): driving gaze is median filtered but not clamped")
        dataset.eval = True
        scene = Scene(dataset, gaussians, shuffle=False)
        identity_index_path = os.path.join(dataset.model_path, "identity_index.txt")
        if os.path.exists(identity_index_path):
            with open(identity_index_path) as f:
                gaussians.identity_index = int(f.read().strip())
        motion_net = MotionNetwork(args=dataset).cuda()
        motion_net.load_state_dict(motion_params, strict=False)
        motion_net.eval()
        gaussians.restore(model_params, None)
        if not args.keep_driving_identity and gaussians.apply_identity(with_shape=args.identity_shape):
            print("[render] avatar bound to its adaptation canonical mesh"
                  + (" + FLAME shape" if args.identity_shape else ""))
        elif not args.keep_driving_identity and not args.canonical_from:
            print("[render] WARNING: checkpoint does not store its adaptation canonical mesh (adapted with an older "
                  "version). When driving with another clip, pass --canonical_from data/<adaptation clip>.")
        deform_code_path = os.path.join(dataset.model_path, "deform_code.pth")
        deform_dec_path = os.path.join(dataset.model_path, "deform_decoder.pth")
        if os.path.exists(deform_code_path) and os.path.exists(deform_dec_path):
            from scene.deform_decoder import ExprDeformDecoder
            dd = ExprDeformDecoder().cuda()
            dd.load_state_dict(torch.load(deform_dec_path, map_location="cuda:0", weights_only=False))
            dd.eval()
            gaussians.deform_decoder = dd
            gaussians.deform_code = torch.load(deform_code_path, map_location="cuda:0", weights_only=False).cuda()
        if args.canonical_from:
            can = np.load(os.path.join(args.canonical_from, "vertices.npy"))
            b = gaussians.flame_binding
            b.canonical_mesh_vertices = gaussians.flame_wrapper.pad_canonical(
                torch.tensor(can).float().to(b.canonical_mesh_vertices.device))
            b._canonical_frames = None
            b._update_canonical_points()
            print(f"canonical mesh overridden from {args.canonical_from}")
        clip_tf = None
        cc_path = os.path.join(dataset.model_path, "clip_colors.pth")
        if os.path.exists(cc_path):
            cc = torch.load(cc_path, map_location="cuda", weights_only=False)
            names = [os.path.basename(os.path.normpath(c)) for c in cc["clips"]]
            drv = os.path.basename(os.path.normpath(dataset.source_path))
            if drv in names and names.index(drv) > 0:
                clip_tf = cc["params"][names.index(drv)].cuda()
                print(f"[render] applying the colour compensation of adaptation clip '{drv}'")
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        views = scene.getTestCameras()
        start = max(0, int(args.eval_from or 0))
        stop = len(views) if int(args.eval_to) < 0 else min(len(views), int(args.eval_to))
        ident = getattr(gaussians, "identity_feature", None)
        code = None
        if getattr(motion_net, "individual_dim", 0) > 0:
            code = motion_net.identity_code(int(getattr(gaussians, "identity_index", 0)))

        torch.cuda.synchronize(); _T.append(("load model+scene", _time.perf_counter()))
        def _au_rows(view, rows):
            a = view.talking_dict.get("au_window", view.talking_dict["au_features"])
            return a.reshape(-1, a.shape[-1]).expand(rows, -1) if a.reshape(-1, a.shape[-1]).shape[0] == 1 else a
        rows = views[0].talking_dict["auds"].shape[0]
        view_preds = [None] * len(views)
        exps, jaws = [], []
        CH = 128
        for c0 in range(0, len(views), CH):
            vv = views[c0:c0 + CH]
            A = torch.cat([v.talking_dict["auds"] for v in vv], 0).cuda()
            U = torch.cat([_au_rows(v, rows) for v in vv], 0).cuda()
            out = motion_net(A, c=code, au=U, identity=ident)
            for k, v in enumerate(vv):
                view_preds[c0 + k] = {key: (val[k * rows:(k + 1) * rows] if torch.is_tensor(val) and val.dim() > 0
                                            and val.shape[0] == len(vv) * rows else val) for key, val in out.items()}
            exps.append(out["flame_exp"].reshape(len(vv), rows, -1)[:, rows // 2].float().cpu().numpy())
            jaws.append(out["flame_jaw"].reshape(len(vv), rows, -1)[:, rows // 2].float().cpu().numpy())
        exps, jaws = np.concatenate(exps), np.concatenate(jaws)
        aux_paths = [] if args.no_aux_motion else sorted(glob.glob(os.path.join(dataset.model_path, "aux_motion_*.pth")))
        all_e = [exps]
        if aux_paths and not args.drive_with_gt:
            from types import SimpleNamespace
            from utils.audio_features import AUDIO_FEATURE_FILES, feature_file, feature_path
            from utils.audio_utils import get_audio_features
            af = getattr(args, "audio_file", "") or ""
            all_j = [jaws]
            for ap_ in aux_paths:
                aux = torch.load(ap_, map_location="cuda", weights_only=False)
                ext, alag = aux["audio_extractor"], int(aux["audio_lag"])
                if af:
                    fpath = getattr(args, "aux_audio_file", None)
                    if not fpath:
                        cands = [f"{os.path.splitext(af)[0]}_{ext}.npy"]
                        if os.path.basename(af) in AUDIO_FEATURE_FILES.values():
                            cands.append(os.path.join(os.path.dirname(af), feature_file(ext)))
                        fpath = next((c for c in cands if os.path.exists(c)), None)
                    if not fpath or not os.path.exists(fpath):
                        raise SystemExit(
                            f"[render] {os.path.basename(ap_)} (motion ensemble) needs {ext} features of the novel "
                            f"audio: extract them from the same wav (tools/extract_"
                            f"{'wav2vec2' if ext == 'wav2vec2' else 'whisper'}_features.py --wav <new.wav> --output "
                            f"<file>) and pass --aux_audio_file <file>, or render without the ensemble (--no_aux_motion).")
                    n_main = np.load(af, mmap_mode="r").shape[0]
                    n_aux = np.load(fpath, mmap_mode="r").shape[0]
                    if n_aux != n_main:
                        raise SystemExit(f"[render] {fpath} has {n_aux} frames but --audio_file has {n_main}: not the "
                                         f"same audio (pass the matching --aux_audio_file, or --no_aux_motion).")
                else:
                    fpath = feature_path(dataset.source_path, ext)
                    if not os.path.exists(fpath):
                        print(f"[render] WARNING: {os.path.basename(ap_)} skipped, no {fpath} (extract it with "
                              f"tools/extract_{'wav2vec2' if ext == 'wav2vec2' else 'whisper'}_features.py); "
                              f"rendering with the main motion network only")
                        continue
                feats = torch.from_numpy(np.load(fpath)).float().permute(0, 2, 1)
                anet = MotionNetwork(args=SimpleNamespace(audio_extractor=ext)).cuda()
                anet.load_state_dict(aux["state_dict"], strict=False)
                anet.eval()
                acode = anet.identity_code(0) if getattr(anet, "individual_dim", 0) > 0 else None
                ae, aj = [], []
                for c0 in range(0, len(views), CH):
                    idx = range(c0, min(c0 + CH, len(views)))
                    W = torch.stack([get_audio_features(feats, 2, (vi if af else int(views[vi].talking_dict["frame_id"])) - alag)
                                     for vi in idx], 0).cuda()
                    nr = W.shape[1]
                    U = torch.stack([_au_rows(views[vi], nr) for vi in idx], 0).cuda()
                    pr = anet(W.reshape(-1, *W.shape[2:]), c=acode, au=U.reshape(-1, U.shape[-1]), identity=ident)
                    ae.append(pr["flame_exp"].reshape(len(idx), nr, -1)[:, nr // 2].float().cpu().numpy())
                    aj.append(pr["flame_jaw"].reshape(len(idx), nr, -1)[:, nr // 2].float().cpu().numpy())
                all_e.append(np.concatenate(ae)); all_j.append(np.concatenate(aj))
                print(f"[render] motion ensemble: + {os.path.basename(ap_)} ({ext}, audio lag {alag})")
            exps, jaws = np.mean(all_e, 0), np.mean(all_j, 0)
        raw_exps, raw_jaws = exps.copy(), jaws.copy()
        mode = getattr(args, "motion_norm", "auto") or "auto"
        if args.articulation_gain and mode == "auto":
            mode = "gain"
        elif args.articulation_gain and mode in ("identity", "none") and not args.drive_with_gt:
            raise SystemExit(f"[render] --articulation_gain has no effect under --motion_norm {mode}: use "
                             f"--motion_norm gain (or auto) with it, or drop it")
        P_up = motion_net.upper_face_proj.detach().cpu().numpy() if motion_net.is_au_decoupled else None
        if mode == "identity" and not args.drive_with_gt and P_up is None:
            raise SystemExit("[render] --motion_norm identity needs an AU-decoupled model (released / --au_mode "
                             "legacy checkpoints have no upper/lower-face split): use --motion_norm gain or none")
        if mode in ("auto", "identity") and not args.drive_with_gt and P_up is not None:
            st = identity_motion_stats(dataset.model_path, adapt_src, P_up)
            if st is not None:
                exps, jaws, slope, scale = match_identity_motion(exps, jaws, st, P_up)
                print(f"[render] motion matched to the identity's distribution (jaw-opening slope {slope:.2f}, "
                      f"lower-face scale {scale:.2f}; {st['n']} adaptation frames)")
                mode = "done"
            elif mode == "identity":
                raise SystemExit("[render] --motion_norm identity: no adaptation flame_params.npz / motion_stats.json")
        if mode == "done" or mode == "none":
            gain = "1,1"
        elif args.articulation_gain:
            gain = args.articulation_gain
        elif not motion_net.is_au_decoupled:
            gain = "1,1"
        else:
            gain = DEFAULT_GAIN_ENSEMBLE if (aux_paths and len(all_e) > 1) else DEFAULT_GAIN_SINGLE
        gj, ge = (float(x) for x in gain.split(","))
        if not args.drive_with_gt and mode != "done":
            print(f"[render] articulation gain jaw {gj:g}, lower face {ge:g}"
                  + ("" if motion_net.is_au_decoupled else
                     " (model is not AU-decoupled: off by default; a gain given here scales the whole expression)"))
        if not args.drive_with_gt and (gj != 1.0 or ge != 1.0):
            jaws = jaws.mean(0, keepdims=True) + gj * (jaws - jaws.mean(0, keepdims=True))
            P = motion_net.upper_face_proj.detach().cpu().numpy() if motion_net.is_au_decoupled else np.zeros((exps.shape[1],) * 2)
            low = exps @ (np.eye(exps.shape[1]) - P).T
            exps = exps + (ge - 1.0) * (low - low.mean(0, keepdims=True))
        sigma = getattr(args, "smooth_sigma", None)
        if sigma is None:
            sigma = 1.0
            _setup = os.path.join(args.model_path, "adaptation_setup.json")
            if os.path.exists(_setup):
                import json as _json
                with open(_setup) as _f:
                    sigma = 0.75 if int(_json.load(_f).get("motion_steps", 0) or 0) > 0 else 1.0
        sigma = float(sigma or 0)
        smoothed = None
        if not args.drive_with_gt:
            from scipy.ndimage import gaussian_filter1d
            sigma_exp = sigma if getattr(args, "smooth_sigma_exp", None) is None else float(args.smooth_sigma_exp)
            if sigma_exp > 0:
                exps = gaussian_filter1d(exps, sigma=sigma_exp, axis=0)
            if sigma > 0:
                jaws = gaussian_filter1d(jaws, sigma=sigma, axis=0)
            if args.eyelids == "frame":
                from utils.flame_eyelids import blink_direction
                from utils.flame_regions import upper_face_projector
                fw = gaussians.flame_wrapper
                _cache = os.path.join(dataset.model_path, "eyelid_basis.npz")
                if os.path.exists(_cache):
                    _u, _w = np.load(_cache)["u"], np.load(_cache)["w"]
                else:
                    with torch.enable_grad():
                        _u, _w = blink_direction(fw.flame_model, upper_proj=upper_face_projector(exps.shape[1])[0])
                    _u, _w = _u.cpu().numpy(), _w.cpu().numpy()
                    np.savez(_cache, u=_u, w=_w)
                drv = np.stack([np.asarray(fw.all_flame_params[int(v.talking_dict["frame_id"])]["expression"],
                                           np.float32).reshape(-1)[:exps.shape[1]] for v in views])
                lid_pred, lid_drv = exps @ _w, drv @ _w
                lid_rng, lid_adapt = None, []
                for _src, _ids in _adaptation_clips(dataset.model_path, adapt_src):
                    _fpz = os.path.join(_src, "flame_params.npz")
                    if os.path.exists(_fpz):
                        _ap = load_drive_params(_fpz)[0]
                        lid_adapt += [float(np.asarray(_ap[i]["expression"], np.float32).reshape(-1)[:exps.shape[1]] @ _w)
                                      for i in _ids if i in _ap]
                if len(lid_adapt) >= 25 and not getattr(args, "no_eyelid_clamp", False):
                    lid_rng = (min(lid_adapt) - float(getattr(args, "eyelid_open_margin", EYELID_MARGIN)),
                               max(lid_adapt) + EYELID_MARGIN)
                    n_out = int(((lid_drv < lid_rng[0]) | (lid_drv > lid_rng[1])).sum())
                    lid_drv = np.clip(lid_drv, *lid_rng)
                exps = (exps + (lid_drv - lid_pred)[:, None] * _u[None]).astype(np.float32)
                print(f"[render] eyelids copied from the driving frames (closure coordinate: predicted "
                      f"{lid_pred.min():.2f}..{lid_pred.max():.2f}, driving {lid_drv.min():.2f}..{lid_drv.max():.2f}"
                      + (f"; clamped to the adaptation range {lid_rng[0]:.2f}..{lid_rng[1]:.2f}, {n_out} frames outside)"
                         if lid_rng else "; no adaptation range available, not clamped)"))
            if getattr(args, "drive_file", None):
                _d = np.load(args.drive_file)
                exps, jaws = _d["exp"].astype(np.float32), _d["jaw"].astype(np.float32)
                assert len(exps) == len(views), f"--drive_file has {len(exps)} rows, the scene {len(views)} frames"
                print(f"[render] driving with {args.drive_file}")
            smoothed = (torch.tensor(exps).float().cuda(), torch.tensor(jaws).float().cuda())
        torch.cuda.synchronize(); _T.append(("motion prediction (+ensemble)", _time.perf_counter()))
        mouth_seq = None
        if smoothed is not None and sigma > 0 and not getattr(gaussians, "intraoral_rigid", False):
            from scipy.ndimage import gaussian_filter1d
            collected = []
            _R = int(4 * sigma + 0.5) + 1
            mouth_lo = max(0, int(args.eval_from or 0) - _R)
            mouth_hi = min(len(views), (len(views) if int(args.eval_to) < 0 else int(args.eval_to)) + _R)
            for vi in range(mouth_lo, mouth_hi):
                view = views[vi]
                mp = render_motion(view, gaussians, motion_net, pipe, background,
                                   drive_params=(smoothed[0][vi], smoothed[1][vi]),
                                   return_motion_only=True, preds=view_preds[vi])["mouth_pred"]
                collected.append(mp)
            if collected and collected[0] is not None:
                keys = ["d_mu", "d_rot", "d_scale"]
                stacks = {k: np.stack([c[k].float().cpu().numpy() for c in collected]) for k in keys}
                mouth_seq = {k: torch.tensor(gaussian_filter1d(v, sigma=sigma, axis=0)).float().cuda()
                             for k, v in stacks.items()}
        torch.cuda.synchronize(); _T.append(("mouth offsets", _time.perf_counter()))

        bg_dir = os.path.join(dataset.source_path, "bg_imgs")
        if args.bg_mode == "gt" and not os.path.isdir(bg_dir):
            print("[render] bg_mode gt requested but the scene has no bg_imgs/; using white.")
            args.bg_mode = "white"
        bg_plate = None
        if args.bg_mode == "gt":
            os.makedirs(os.path.join(dataset.model_path, "test"), exist_ok=True)
            import hashlib
            key = hashlib.sha1(os.path.realpath(os.path.join(dataset.source_path, "bg_imgs")).encode()).hexdigest()[:10]
            plate_path = os.path.join(dataset.model_path, "test",
                                      f"bg_plate_{os.path.basename(dataset.source_path.rstrip('/'))}_{key}.png")
            missing = [views[vi].talking_dict['img_id'] for vi in range(start, stop)
                       if not os.path.exists(os.path.join(bg_dir, f"{views[vi].talking_dict['img_id']}.jpg"))]
            if missing:
                raise FileNotFoundError(f"[render] bg_mode real: {len(missing)} frame(s) have no bg_imgs/<id>.jpg "
                                        f"(e.g. {missing[:5]}); use --bg_mode white")
            bg_plate = torch.from_numpy(_estimate_bg_plate(dataset.source_path, plate_path)).permute(2, 0, 1).cuda()
        bc_plate = None
        rframes, gframes, masks = [], [], []
        t_gs = 0.0
        from concurrent.futures import ThreadPoolExecutor
        _H0 = int(views[start].image_height) if stop > start else 0

        _curves = {}

        def _frame_io(vi):
            img_id = views[vi].talking_dict["img_id"]
            src = None
            if args.bg_mode == "gt":
                src = imageio.imread(os.path.join(bg_dir, f"{img_id}.jpg"))
            row = _neck_row_from_landmarks(dataset.source_path, img_id, _H0)
            if getattr(args, "torso_boundary", "curve") == "curve":
                _curves[vi] = _torso_curve(dataset.source_path, img_id, _H0, _H0, row, float(getattr(args, "torso_side_lift", 0.2)))
            return src, row, _load_matte(dataset.source_path, img_id, _H0)
        _pool, _futs, _AHEAD = ThreadPoolExecutor(8), {}, 32
        for _vi in range(start, min(start + _AHEAD, stop)):
            _futs[_vi] = _pool.submit(_frame_io, _vi)
        torch.cuda.synchronize(); _T.append(("background plate (first run only, then cached)", _time.perf_counter()))
        for vi in range(start, stop):
            view = views[vi]
            dp = (smoothed[0][vi], smoothed[1][vi]) if smoothed is not None else None
            mp = {k: v[vi - mouth_lo] for k, v in mouth_seq.items()} if mouth_seq is not None else None
            torch.cuda.synchronize(); _t0 = _time.perf_counter()
            pkg = render_motion(view, gaussians, motion_net, pipe, background,
                                drive_with_gt=args.drive_with_gt, drive_params=dp, mouth_params=mp,
                                preds=view_preds[vi])
            torch.cuda.synchronize(); t_gs += _time.perf_counter() - _t0
            _src, _nk, _matte = _futs.pop(vi).result()
            if vi + _AHEAD < stop:
                _futs[vi + _AHEAD] = _pool.submit(_frame_io, vi + _AHEAD)
            rgb = pkg["render"]
            alpha = pkg["alpha"].clamp(0, 1)
            if clip_tf is not None:
                eye = torch.eye(3, device=rgb.device, dtype=rgb.dtype)
                rgb = torch.einsum('ij,jhw->ihw', eye + clip_tf[:9].view(3, 3), rgb) + clip_tf[9:12].view(3, 1, 1) * alpha
            rgb = rgb.clamp(0, 1)
            H = rgb.shape[-2]
            gt = view.original_image.cuda().float()
            if gt.max() > 1.5:
                gt = gt / 255.0
            gt = gt.clamp(0, 1)
            if args.bg_mode == "gt":
                gt = torch.from_numpy(_src.astype(np.float32) / 255.0).permute(2, 0, 1).cuda().clamp(0, 1)
                backdrop = bg_plate
                if getattr(args, "frame_background", False):
                    fm = _matte
                    if fm is not None:
                        pure_bg = (fm.to(gt.device) <= 0).float()
                        backdrop = gt * pure_bg + bg_plate * (1 - pure_bg)
            else:
                backdrop = view.background.cuda().float() / 255.0
                nk_w = _nk
                bc_path = view.talking_dict.get('bg_img_path')
                if nk_w is not None and bc_path and os.path.exists(bc_path):
                    if bc_plate is None:
                        bc_plate = torch.from_numpy(imageio.imread(bc_path)[..., :3].astype(np.float32) / 255.0).permute(2, 0, 1).cuda()
                    backdrop = backdrop.clone()
                    backdrop[:, :nk_w] = bc_plate[:, :nk_w]
            comp = rgb + backdrop * (1.0 - alpha)
            nk = _nk
            if nk is not None:
                band = max(8, int(0.05 * H))
                rows = torch.arange(H, device=comp.device, dtype=comp.dtype)
                _cur = _curves.pop(vi, None)
                if _cur is None:
                    t = ((rows - nk) / (2 * band)).clamp(0, 1)
                    wgt = (1 - (3 * t ** 2 - 2 * t ** 3)).view(1, H, 1)
                else:
                    t = ((rows.view(H, 1) - torch.from_numpy(_cur).to(comp).view(1, -1)) / (2 * band)).clamp(0, 1)
                    wgt = (1 - (3 * t ** 2 - 2 * t ** 3)).unsqueeze(0)
                comp = comp * wgt + gt * (1 - wgt)
            matte = _matte
            if args.trim_matte and matte is not None:
                pm = _trim_matte(matte).to(comp.device)
                comp = comp * pm + backdrop * (1 - pm)
            masks.append(_head_region(matte, view.talking_dict.get("face_mask"), nk, H, curve=_cur if nk is not None else None))
            rframes.append((comp.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
            gframes.append((gt.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))

        _pool.shutdown()
        out = os.path.join(dataset.model_path, "test")
        os.makedirs(out, exist_ok=True)
        name = args.out_name
        rpath = os.path.join(out, f"rendered_{name}.mp4")
        gpath = os.path.join(out, f"gt_{name}.mp4")
        torch.cuda.synchronize(); _T.append(("render + composite + frame I/O", _time.perf_counter()))
        imageio.mimwrite(rpath, rframes, fps=25, quality=9, macro_block_size=1)
        imageio.mimwrite(gpath, gframes, fps=25, quality=9, macro_block_size=1)
        np.savez_compressed(os.path.join(out, f"masks_{name}.npz"),
                            masks=np.packbits(np.stack(masks), axis=-1), shape=np.array(np.stack(masks).shape))
        np.savez_compressed(os.path.join(out, f"motion_{name}.npz"), exp=exps, jaw=jaws, exp_raw=raw_exps,
                            jaw_raw=raw_jaws, frame_ids=np.array([int(v.talking_dict["frame_id"]) for v in views]),
                            start=start, stop=stop)
        wav = args.wav
        if wav is None:
            af = getattr(args, "audio_file", "") or ""
            cand = [os.path.splitext(af)[0] + ".wav"] if af else [os.path.join(dataset.source_path, "aud.wav")]
            wav = next((c for c in cand if os.path.exists(c)), None)
        if wav:
            _mux(rpath, wav, os.path.join(out, f"rendered_{name}_audio.mp4"), start_frame=start)
            side = os.path.join(out, f"side_{name}.mp4")
            imageio.mimwrite(side, [np.concatenate([g, r], axis=1) for g, r in zip(gframes, rframes)],
                             fps=25, quality=9, macro_block_size=1)
            if _mux(side, wav, os.path.join(out, f"side_{name}_audio.mp4"), start_frame=start):
                os.remove(side)
        torch.cuda.synchronize(); _T.append(("write videos", _time.perf_counter()))
        print("saved", len(rframes), "frames to", out)
        nfr = max(len(rframes), 1)
        print("[render] timing: " + ", ".join(f"{n} {(_T[i][1] - _T[i - 1][1]):.1f}s"
                                              for i, (n, _) in enumerate(_T) if i > 0)
              + f" | {nfr} frames: motion {1000 * (_T[2][1] - _T[1][1]) / nfr:.1f} ms/frame, mouth "
              f"{1000 * (_T[3][1] - _T[2][1]) / nfr:.1f} ms/frame, 3DGS render {1000 * t_gs / nfr:.1f} ms/frame, "
              f"compositing + frame I/O {1000 * (_T[5][1] - _T[4][1] - t_gs) / nfr:.1f} ms/frame")


if __name__ == "__main__":
    main()
