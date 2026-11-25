import glob
import json
import os
import random
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, l2_loss, patchify, ssim, normalize, semantic_emotion_guidance_loss
from gaussian_renderer import render, render_motion
import sys
from scene import Scene, GaussianModel, MotionNetwork
from utils.general_utils import safe_state
from utils.mask_utils import unpack_bool_mask
import lpips
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.normal_utils import depth_to_normal
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.flame_wrapper import SimpleFlameWrapper
from utils.drive_sanitize import drive_convention, gaze_range, raw_drive
from utils.visualize_utils import ImageDebugger, TrainingStatsTracker, FlameDebugger
try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np
from PIL import Image
def matted_target(gt_original, soft_matte, background, white_matted):
    """Supervision image = premultiplied foreground + background * (1 - alpha)."""
    if white_matted:
        fg = (gt_original - (1.0 - soft_matte)).clamp(0.0, 1.0)
    else:
        fg = gt_original * soft_matte
    return fg + background[:, None, None] * (1.0 - soft_matte)


def apply_clip_color(rgb_premult, alpha, params):
    """Per-clip colour compensation on the premultiplied foreground."""
    A = params[:9].view(3, 3)
    b = params[9:12].view(3, 1, 1)
    eye = torch.eye(3, device=rgb_premult.device, dtype=rgb_premult.dtype)
    return torch.einsum('ij,jhw->ihw', eye + A, rgb_premult) + b * alpha


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, mode_long, pretrain_ckpt_path, warm_up_iter, adapt_adain_only=False, adapt_unfreeze_heads=False, au_mode='decoupled', adapt_unfreeze_backbone=False, window_supervision=True, w_vel=1.0, extra_scenes=(), extra_views=None, intraoral='audio', motion_steps=0):
    scene_name = os.path.basename(dataset.model_path)
    loss_history = {scene_name: []}
    testing_iterations = [1] + [i for i in range(0, opt.iterations + 1, 10000)]
    checkpoint_iterations = saving_iterations = [i for i in range(0, opt.iterations + 1, 10000)] + [opt.iterations]
    if opt.densify_from_iter <= warm_up_iter:
        new_densify_from_iter = warm_up_iter + 100
        opt.densify_from_iter = new_densify_from_iter
    if warm_up_iter > 0:
        new_opacity_reset_interval = warm_up_iter
        opt.opacity_reset_interval = new_opacity_reset_interval
    _opr = os.environ.get('EMOTAG_OPACITY_RESET')
    if _opr:
        opt.opacity_reset_interval = int(_opr)
    opt.densify_until_iter = opt.iterations - 1000
    curriculum_transition_iters = max(1, int((opt.iterations - warm_up_iter) * 0.1))
    curriculum_start_iter = warm_up_iter + 1
    curriculum_end_iter = curriculum_start_iter + curriculum_transition_iters
    first_iter = 0
    motion_net = MotionNetwork(args=dataset).cuda()
    motion_optimizer = torch.optim.AdamW(motion_net.get_params(0.005, 0.0005, include_novel_code=True), betas=(0.9, 0.99), eps=1e-08, weight_decay=0.01)
    _ARTICULATION_GROUPS = ('neural_flame_exp_head', 'neural_flame_jaw_head', 'neural_audio_adain', 'neural_au_adain', 'neural_novel_code')

    def _motion_lr(name):
        if motion_steps > 0 and name in _ARTICULATION_GROUPS:
            return lambda it: 0.1 ** (min(it, motion_steps) / motion_steps)
        return lambda it: 0.1 ** (it / opt.iterations)
    scheduler = torch.optim.lr_scheduler.LambdaLR(motion_optimizer, [_motion_lr(g['name']) for g in motion_optimizer.param_groups])
    os.makedirs(dataset.model_path, exist_ok=True)
    with open(os.path.join(dataset.model_path, 'cfg_args'), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(dataset))))
    flame_params_file = os.path.join(dataset.source_path, 'flame_params.npz')
    if not os.path.exists(flame_params_file):
        sys.exit(f'[adapt] Missing {flame_params_file}. Run the data-preparation pipeline for this scene first.')
    model_center_path = os.path.join(dataset.source_path, 'model_center.npy')
    if not os.path.exists(model_center_path):
        sys.exit(f'[adapt] Missing {model_center_path}. Run the data-preparation pipeline for this scene first.')
    model_center = np.load(model_center_path)
    dataset.model_center = model_center
    flame_wrapper = SimpleFlameWrapper(flame_params_file=flame_params_file, add_teeth=intraoral == 'rigid').cuda()
    gaussians = GaussianModel(dataset)
    gaussians.flame_wrapper = flame_wrapper
    identity_path = os.path.join(dataset.source_path, 'identity_feature.npy')
    if os.path.exists(identity_path):
        gaussians.identity_feature = torch.from_numpy(np.load(identity_path).reshape(-1)).float().cuda()
    scene = Scene(dataset, gaussians)
    if intraoral == 'rigid':
        gaussians.add_intraoral_gaussians(n_teeth=opt.intraoral_teeth, n_cavity=opt.intraoral_cavity,
                                          budget=opt.intraoral_budget)
        if opt.intraoral_contain:
            from utils import intraoral as _intraoral
            print(f'[intraoral] cavity ownership on: {_intraoral.adopt(gaussians)} scene points inside the cavity adopted, '
                  f'{_intraoral.contain(gaussians)} set gaussians moved into the volume')
    elif opt.intraoral_contain:
        print('[adapt] --intraoral_contain has no effect without --intraoral rigid')
    gaussians.training_setup(opt)
    if pretrain_ckpt_path is None or not os.path.exists(pretrain_ckpt_path):
        sys.exit(
            f'[adapt] --pretrain_path is required and must point to an existing multi-identity '
            f'EMA checkpoint (e.g. output/pretrain/chkpnt_ema_face_latest.pth). Got: {pretrain_ckpt_path}'
        )
    pretrain_ckpt = torch.load(pretrain_ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(pretrain_ckpt, (list, tuple)) and len(pretrain_ckpt) == 3:
        motion_params = pretrain_ckpt[0]
    elif isinstance(pretrain_ckpt, (list, tuple)) and len(pretrain_ckpt) == 4:
        motion_params = pretrain_ckpt[1]
    elif isinstance(pretrain_ckpt, dict):
        motion_params = pretrain_ckpt
    else:
        sys.exit(
            f'[adapt] Unrecognized pretrain checkpoint structure at {pretrain_ckpt_path}. '
            f'Expected a 3-tuple EMA checkpoint, a 4-tuple per-scene checkpoint, or a raw state_dict.'
        )
    from utils.audio_features import check_prior_extractor
    motion_params = check_prior_extractor(motion_params, dataset.audio_extractor, pretrain_ckpt_path)
    motion_net.load_state_dict(motion_params)
    flame_wrappers = [flame_wrapper]
    primary_cams = scene.getTrainCameras()
    for c in primary_cams:
        c.talking_dict['scene_idx'] = 0
        c.talking_dict['source_path'] = dataset.source_path
    extra_cams = []
    for k, extra in enumerate(extra_scenes, start=1):
        from scene.dataset_readers import readCamerasFromTransforms
        from utils.camera_utils import cameraList_from_camInfos
        extra = os.path.abspath(extra)
        n_extra = extra_views if extra_views is not None else dataset.N_views
        infos = readCamerasFromTransforms(extra, 'transforms.json', dataset.white_background, '.jpg', '', dataset.audio_extractor,
                                          n_extra, preload=True, audio_lag=getattr(dataset, 'audio_lag', 0))
        cams = cameraList_from_camInfos(infos, 1.0, dataset)
        for c in cams:
            c.talking_dict['scene_idx'] = k
            c.talking_dict['source_path'] = extra
        extra_cams.extend(cams)
        flame_wrappers.append(SimpleFlameWrapper(flame_params_file=os.path.join(extra, 'flame_params.npz'),
                                                 add_teeth=intraoral == 'rigid').cuda())
        print(f'[adapt] extra clip {k}: {extra} ({len(cams)} frames)')
    all_train_cams = list(primary_cams) + extra_cams
    train_cams = all_train_cams
    clip_color, clip_color_opt = None, None
    if extra_cams:
        clip_color = torch.nn.Parameter(torch.zeros(len(flame_wrappers), 12, device='cuda'))
        clip_color_opt = torch.optim.Adam([clip_color], lr=1e-3)
        black_background = torch.zeros(3, device='cuda')
    train_frame_ids = sorted({int(c.talking_dict['frame_id']) for c in primary_cams})
    if au_mode == 'decoupled':
        from utils.flame_regions import upper_face_projector
        au_ref = torch.stack([c.talking_dict['au_features'] for c in train_cams]).mean(0)
        proj, proj_info = upper_face_projector(motion_net.flame_exp_dim)
        motion_net.configure_au_decoupling(au_ref, proj, enabled=True)
        print(f'[adapt] AU decoupling on: reference AU = {au_ref.numpy().round(3).tolist()}, '
              f'upper-face expression sub-space = {proj_info["kept"]} directions')
    motion_net.init_novel_code()
    if adapt_adain_only:
        motion_net.freeze_for_adaptation(unfreeze_heads=adapt_unfreeze_heads, unfreeze_backbone=adapt_unfreeze_backbone)
    gt_tables = []
    for k, fw in enumerate(flame_wrappers):
        _ids = [int(c.talking_dict['frame_id']) for c in all_train_cams if c.talking_dict['scene_idx'] == k]
        fw.isolate_frames(_ids)
        _fp = fw.all_flame_params
        _n = max(_fp) + 1
        _ge = torch.zeros(_n, motion_net.flame_exp_dim, device='cuda')
        _gj = torch.zeros(_n, motion_net.flame_jaw_dim, device='cuda')
        for _fid, _p in _fp.items():
            _ge[_fid] = torch.as_tensor(np.asarray(_p['expression']).reshape(-1)[:motion_net.flame_exp_dim]).float()
            _gj[_fid] = torch.as_tensor(np.asarray(_p['jaw_pose']).reshape(-1)[:motion_net.flame_jaw_dim]).float()
        _mask = torch.zeros(_n, dtype=torch.bool, device='cuda')
        _mask[torch.as_tensor(_ids, device='cuda')] = True
        gt_tables.append((_ge, _gj, _mask, _n))
    with open(os.path.join(dataset.model_path, 'adaptation_frames.txt'), 'w') as _f:
        _f.write(' '.join(str(i) for i in train_frame_ids))
    _clips = [{'source_path': os.path.abspath(dataset.source_path), 'frame_ids': train_frame_ids}]
    for k, extra in enumerate(extra_scenes, start=1):
        _clips.append({'source_path': os.path.abspath(extra),
                       'frame_ids': sorted({int(c.talking_dict['frame_id']) for c in extra_cams
                                            if c.talking_dict['scene_idx'] == k})})
    for _c in _clips:
        _c['drive_convention'] = drive_convention(os.path.join(_c['source_path'], 'flame_params.npz'))
    _convention = _clips[0]['drive_convention']
    print(f'[adapt] FLAME parameter convention of the adaptation clip: {_convention}')
    if any(_c['drive_convention'] != _convention for _c in _clips):
        print('[adapt] WARNING: the adaptation clips do not share one FLAME parameter convention '
              f'({[_c["drive_convention"] for _c in _clips]}): run tools/refine_landmarks.py on all of them or on none')
    _gaze_range = None
    if not raw_drive():
        _gaze_range = gaze_range([(fw.all_flame_params, c['frame_ids']) for fw, c in zip(flame_wrappers, _clips)])
        if _gaze_range is not None:
            for fw in flame_wrappers:
                fw.clamp_gaze(*_gaze_range)
    for _stale in ['motion_stats.json'] + glob.glob(os.path.join(dataset.model_path, 'aux_motion_*.pth')):
        _sp = os.path.join(dataset.model_path, _stale) if not os.path.isabs(_stale) else _stale
        if os.path.exists(_sp):
            os.remove(_sp)
            print(f'[adapt] removed {os.path.basename(_sp)} of an earlier adaptation (refit tools/fit_aux_motion.py)')
    with open(os.path.join(dataset.model_path, 'adaptation_setup.json'), 'w') as _f:
        json.dump({'clips': _clips, 'audio_extractor': dataset.audio_extractor,
                   'audio_lag': int(getattr(dataset, 'audio_lag', 0)), 'au_mode': au_mode,
                   'adapt_adain_only': bool(adapt_adain_only), 'adapt_unfreeze_heads': bool(adapt_unfreeze_heads),
                   'adapt_unfreeze_backbone': bool(adapt_unfreeze_backbone),
                   'window_supervision': bool(window_supervision), 'w_vel': float(w_vel), 'motion_steps': int(motion_steps),
                   'iterations': int(opt.iterations), 'warm_up_iter': int(warm_up_iter),
                   'w_jaw_reg': float(opt.w_jaw_reg), 'w_emotion_kl': float(opt.w_emotion_kl),
                   'w_emotion_score': float(opt.w_emotion_score),
                   'pretrain_path': os.path.abspath(pretrain_ckpt_path) if pretrain_ckpt_path else None,
                   'intraoral': intraoral, 'intraoral_contain': bool(opt.intraoral_contain),
                   'drive_convention': _convention,
                   'gaze_range': [r.tolist() for r in _gaze_range] if _gaze_range is not None else None}, _f, indent=1)
    if intraoral != 'rigid':
        try:
            mouth_idx_path = os.path.join(dataset.source_path, 'mouth_point_indices.npy')
            mouth_indices_np = np.load(mouth_idx_path)
            if mouth_indices_np.ndim > 1:
                mouth_indices_np = mouth_indices_np.reshape(-1)
            N = gaussians.get_xyz.shape[0]
            mouth_indices = torch.from_numpy(mouth_indices_np).long().clamp_(0, max(0, N - 1))
            mouth_mask = torch.zeros(N, dtype=torch.bool, device='cuda')
            if mouth_indices.numel() > 0:
                mouth_mask[mouth_indices.to(mouth_mask.device)] = True
            gaussians.mouth_indices = mouth_indices.to('cuda')
            gaussians.mouth_mask = mouth_mask
            gaussians.freeze_mouth_set = True
            gaussians.mouth_budget = int(opt.mouth_budget_factor * max(1, int(mouth_mask.sum().item())))
        except Exception as e:
            sys.exit(f'[adapt] Failed to load mouth_point_indices.npy for this scene: {e}')
    if checkpoint:
        model_params, motion_params, motion_optimizer_params, first_iter = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)
        motion_net.load_state_dict(motion_params)
        try:
            motion_optimizer.load_state_dict(motion_optimizer_params)
        except ValueError as exc:
            print(f'[adapt] motion optimizer state not restored ({exc}); continuing with a fresh optimizer state')
    if not mode_long and os.environ.get('EMOTAG_ADAPT_SH2') != '1':
        gaussians.max_sh_degree = 1
    deform_optimizer = None
    if os.environ.get('EMOTAG_DEFORM') == '1':
        from scene.deform_decoder import ExprDeformDecoder
        deform_ckpt = os.environ.get('EMOTAG_DEFORM_CKPT') or \
            os.path.join(os.path.dirname(pretrain_ckpt_path), 'chkpnt_deform_ema_latest.pth')
        if not os.path.exists(deform_ckpt):
            sys.exit(f'[adapt] EMOTAG_DEFORM=1 but no deformation prior at {deform_ckpt}')
        dd = ExprDeformDecoder().cuda()
        dd.load_state_dict(torch.load(deform_ckpt, map_location='cuda:0', weights_only=False)[0])
        for p in dd.parameters():
            p.requires_grad_(False)
        dd.eval()
        gaussians.deform_decoder = dd
        init_code = dd.id_codes.detach().mean(dim=0, keepdim=True).clone()
        gaussians.deform_code = torch.nn.Parameter(init_code.cuda())
        deform_optimizer = torch.optim.AdamW([gaussians.deform_code], lr=5e-3, betas=(0.9, 0.99))
        print('loaded expression deformation prior; fitting identity code')
    bg_color = [0, 1, 0]
    fixed_background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
    background = fixed_background
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    _matte_cache = {}
    _white_matted_by_scene = {}
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), ascii=True, dynamic_ncols=True, desc='Training progress')
    first_iter += 1
    data_list = [scene_name]
    scene_paths = {scene_name: dataset.model_path}
    image_debugger = ImageDebugger(dataset.model_path, data_list, warm_up_iter, scene_paths)
    stats_tracker = TrainingStatsTracker(data_list, dataset.model_path)
    flame_debugger = FlameDebugger(dataset.model_path, debug_save_interval=200)
    scene_list = [scene]
    stats_tracker.initialize_from_scenes(scene_list)
    _own = [0, 0]

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if not viewpoint_stack or iteration == warm_up_iter + 1:
            viewpoint_stack = list(primary_cams) if iteration <= warm_up_iter else list(all_train_cams)
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        background = torch.rand(3, device='cuda') if opt.random_background else fixed_background
        _sidx = viewpoint_cam.talking_dict.get('scene_idx', 0)
        gaussians.flame_wrapper = flame_wrappers[_sidx]
        _src = viewpoint_cam.talking_dict.get('source_path', dataset.source_path)
        current_frame_id = viewpoint_cam.image_name
        stats_tracker.record_training(scene_name, current_frame_id)
        if iteration - 1 == debug_from:
            pipe.debug = True
        face_mask = torch.as_tensor(unpack_bool_mask(viewpoint_cam.talking_dict['face_mask'])).cuda()
        hair_mask = torch.as_tensor(unpack_bool_mask(viewpoint_cam.talking_dict['hair_mask'])).cuda()
        head_mask = face_mask
        soft_matte = None
        _mp = os.path.join(_src, 'matte', f"{viewpoint_cam.talking_dict['img_id']}.png")
        if os.path.exists(_mp):
            _mid = (_sidx, viewpoint_cam.talking_dict['img_id'])
            if _mid in _matte_cache:
                soft_matte = _matte_cache[_mid]
            else:
                _m = np.array(Image.open(_mp)).astype(np.float32) / 255.0
                if _m.ndim == 3:
                    _m = _m[..., 0]
                soft_matte = torch.from_numpy(_m).float().unsqueeze(0)
                _matte_cache[_mid] = soft_matte
            soft_matte = soft_matte.cuda()
            if _sidx not in _white_matted_by_scene:
                _bgpx = soft_matte[0] < 0.02
                _go = viewpoint_cam.original_image.cuda() / 255.0
                _white_matted_by_scene[_sidx] = bool(_bgpx.any() and _go[:, _bgpx].mean() > 0.97)
                print(f'[adapt] clip {_sidx}: gt_imgs matted to white: {_white_matted_by_scene[_sidx]} -> supervising the un-premultiplied foreground')
        _white_matted = _white_matted_by_scene.get(_sidx, False)
        if iteration <= warm_up_iter:
            if iteration == 1:
                print('Starting static warm-up stage.')
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            rendered_image = render_pkg['render']
            alpha = render_pkg['alpha']
            viewspace_point_tensor = render_pkg['viewspace_points']
            visibility_filter = render_pkg['visibility_filter']
            radii = render_pkg['radii']
            gt_image_original = viewpoint_cam.original_image.cuda() / 255.0
            gt_image = matted_target(gt_image_original, soft_matte, background, _white_matted) \
                if soft_matte is not None else \
                (gt_image_original * head_mask + background[:, None, None] * ~head_mask)
            lambda_ssim = opt.lambda_dssim
            photo_l1_loss = l1_loss(rendered_image, gt_image)
            photo_ssim_loss = 1.0 - ssim(rendered_image, gt_image)
            loss = (1.0 - lambda_ssim) * photo_l1_loss + lambda_ssim * photo_ssim_loss
        else:
            if iteration == warm_up_iter + 1:
                print('Starting dynamic motion stage.')
            if iteration < curriculum_start_iter:
                curriculum_weight = 0.0
            elif iteration >= curriculum_end_iter:
                curriculum_weight = 1.0
            else:
                curriculum_weight = min(1.0, max(0.0, (iteration - curriculum_start_iter) / curriculum_transition_iters))
            if clip_color is not None and _sidx > 0:
                render_pkg = render_motion(viewpoint_cam, gaussians, motion_net, pipe, black_background, debug_viz=iteration == 1, curriculum_weight=curriculum_weight, drive_with_gt=True)
                render_pkg['render'] = apply_clip_color(render_pkg['render'], render_pkg['alpha'], clip_color[_sidx]) \
                    + background[:, None, None] * (1.0 - render_pkg['alpha'])
            else:
                render_pkg = render_motion(viewpoint_cam, gaussians, motion_net, pipe, background, debug_viz=iteration == 1, curriculum_weight=curriculum_weight, drive_with_gt=True)
            rendered_image = render_pkg['render']
            alpha = render_pkg['alpha']
            viewspace_point_tensor = render_pkg['viewspace_points']
            visibility_filter = render_pkg['visibility_filter']
            radii = render_pkg['radii']
            pred_exp = render_pkg['predicted_flame_exp']
            pred_jaw = render_pkg['predicted_flame_jaw']
            gt_exp = render_pkg['gt_flame_exp']
            gt_pose = render_pkg['gt_flame_pose']
            gt_jaw = gt_pose[:, 3:]
            pred_exp_flat = pred_exp.squeeze()
            pred_jaw_flat = pred_jaw.squeeze()
            gt_exp_flat = gt_exp.squeeze()[:50]
            gt_jaw_flat = gt_jaw.squeeze()
            gt_image_original = viewpoint_cam.original_image.cuda() / 255.0
            gt_image = matted_target(gt_image_original, soft_matte, background, _white_matted) \
                if soft_matte is not None else \
                (gt_image_original * head_mask + background[:, None, None] * ~head_mask)
            lambda_ssim = opt.lambda_dssim
            photo_l1_loss = l1_loss(rendered_image, gt_image)
            photo_ssim_loss = 1.0 - ssim(rendered_image, gt_image)
            rendering_loss = (1.0 - lambda_ssim) * photo_l1_loss + lambda_ssim * photo_ssim_loss
            if opt.w_lpips > 0:
                if not hasattr(training, '_lpips_fn'):
                    training._lpips_fn = lpips.LPIPS(net='alex').eval().cuda()
                    for p in training._lpips_fn.parameters():
                        p.requires_grad_(False)
                lpips_loss = training._lpips_fn(rendered_image.unsqueeze(0), gt_image.unsqueeze(0), normalize=True).squeeze()
                rendering_loss = rendering_loss + opt.w_lpips * lpips_loss
            if opt.w_mouth_crop > 0:
                mx0, mx1, my0, my1 = viewpoint_cam.talking_dict['lips_rect']
                mx0, mx1 = max(0, int(mx0)), min(rendered_image.shape[1], int(mx1))
                my0, my1 = max(0, int(my0)), min(rendered_image.shape[2], int(my1))
                if mx1 - mx0 > 8 and my1 - my0 > 8:
                    rc = rendered_image[:, mx0:mx1, my0:my1]
                    gc = gt_image[:, mx0:mx1, my0:my1]
                    mouth_loss = l1_loss(rc, gc)
                    if opt.w_lpips > 0:
                        rcu = F.interpolate(rc.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False)
                        gcu = F.interpolate(gc.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False)
                        mouth_loss = mouth_loss + 0.2 * training._lpips_fn(rcu, gcu, normalize=True).squeeze()
                    rendering_loss = rendering_loss + opt.w_mouth_crop * mouth_loss
            if window_supervision and 'window_frame_ids' in viewpoint_cam.talking_dict:
                gt_exp_all, gt_jaw_all, train_frame_mask, _n_all = gt_tables[_sidx]
                wids = torch.as_tensor(viewpoint_cam.talking_dict['window_frame_ids'], device='cuda')
                inside = (wids >= 0) & (wids < _n_all)
                valid = torch.zeros_like(inside)
                valid[inside] = train_frame_mask[wids[inside]]
                _aidx = wids - int(getattr(dataset, 'audio_lag', 0))
                valid &= (_aidx >= 0) & (_aidx < _n_all)
                wids_c = wids.clamp(0, _n_all - 1)
                pew = render_pkg['predicted_flame_exp_window']
                pjw = render_pkg['predicted_flame_jaw_window']
                gew, gjw = gt_exp_all[wids_c], gt_jaw_all[wids_c]
                vf = valid.float().unsqueeze(-1)
                nv = vf.sum().clamp_min(1.0)
                flame_exp_loss = (((pew - gew) ** 2) * vf).sum() / (nv * pew.shape[-1])
                flame_jaw_loss = (((pjw - gjw) ** 2) * vf).sum() / (nv * pjw.shape[-1])
                pair = (valid[1:] & valid[:-1]).float().unsqueeze(-1)
                npair = pair.sum().clamp_min(1.0)
                vel_exp = ((((pew[1:] - pew[:-1]) - (gew[1:] - gew[:-1])) ** 2) * pair).sum() / (npair * pew.shape[-1])
                vel_jaw = ((((pjw[1:] - pjw[:-1]) - (gjw[1:] - gjw[:-1])) ** 2) * pair).sum() / (npair * pjw.shape[-1])
                flame_exp_loss = flame_exp_loss + w_vel * vel_exp
                flame_jaw_loss = flame_jaw_loss + w_vel * vel_jaw
            else:
                flame_exp_loss = F.mse_loss(pred_exp_flat, gt_exp_flat)
                flame_jaw_loss = F.mse_loss(pred_jaw_flat, gt_jaw_flat)
            flame_reg_loss = flame_exp_loss + opt.w_jaw_reg * flame_jaw_loss
            w_rendering = 1.0
            w_flame_reg = 0.1
            loss = w_rendering * rendering_loss + w_flame_reg * flame_reg_loss
            emo_target = viewpoint_cam.talking_dict.get('emotion', None)
            if emo_target is not None:
                seg_loss, _, _ = semantic_emotion_guidance_loss(render_pkg.get('emotion_logits'), render_pkg.get('gate'), emo_target, viewpoint_cam.talking_dict.get('emotion_score', 0.0), w_kl=opt.w_emotion_kl, w_score=opt.w_emotion_score)
                loss = loss + seg_loss
            normal_gt = viewpoint_cam.talking_dict.get('normal', None)
            depth_gt = viewpoint_cam.talking_dict.get('depth', None)
            if normal_gt is not None or depth_gt is not None:
                try:
                    geo_mask = head_mask[:1]
                    if depth_gt is not None:
                        rendered_depth = render_pkg['depth']
                        depth_gt = depth_gt.to(rendered_depth.device).reshape(rendered_depth.shape)
                        depth_loss = l1_loss(geo_mask * normalize(rendered_depth), geo_mask * normalize(depth_gt))
                        loss = loss + opt.w_geo_depth * depth_loss
                    if normal_gt is not None:
                        rendered_normal = render_pkg['normal']
                        normal_gt = normal_gt.to(rendered_normal.device).reshape(rendered_normal.shape)
                        normal_loss = l1_loss(geo_mask * rendered_normal, geo_mask * normal_gt)
                        loss = loss + opt.w_geo_normal * normal_loss
                except Exception as geo_exc:
                    if not getattr(training, '_geo_warned', False):
                        print(f'[warn] Skipping geometric loss (shape mismatch?): {geo_exc}')
                        training._geo_warned = True
            flame_debugger.record_flame_params(iteration, scene_name, current_frame_id, pred_exp, pred_jaw, gt_exp, gt_jaw, rendering_loss, flame_reg_loss)
        if (opt.lambda_xyz > 0 or opt.lambda_scale > 0) and gaussians.flame_binding is not None:
            from scene.flame_gaussian_model import binding_mode
            vis = visibility_filter
            if binding_mode() == 'hybrid':
                k_pt = gaussians.flame_binding.canonical_face_frames()[2][:, 0][gaussians.flame_binding.face_indices] + 1e-8
                local_off = gaussians._xyz.norm(dim=1) / k_pt
                local_scale = gaussians.scaling_activation(gaussians._scaling) / k_pt.unsqueeze(1)
            else:
                local_off = gaussians._xyz.norm(dim=1)
                local_scale = gaussians.scaling_activation(gaussians._scaling)
            if opt.lambda_xyz > 0:
                vis_xyz = vis
                if gaussians.intraoral_rigid and gaussians.mouth_mask is not None:
                    vis_xyz = vis & ~gaussians.mouth_mask
                loss = loss + opt.lambda_xyz * F.relu(local_off[vis_xyz] - opt.threshold_xyz).mean()
            if opt.lambda_scale > 0:
                loss = loss + opt.lambda_scale * F.relu(local_scale[vis] - opt.threshold_scale).norm(dim=1).mean()
        loss.backward()
        gaussians.flame_wrapper = flame_wrappers[0]
        iter_end.record()
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                if iteration <= warm_up_iter:
                    progress_bar.set_postfix_str(f'warmup loss={ema_loss_for_log:.6f}')
                else:
                    progress_bar.set_postfix_str(f'dynamic loss={ema_loss_for_log:.6f}')
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            if iteration == 1:
                image_debugger.save_debug_comparison(gt_image_original, head_mask, gt_image, background)
            image_debugger.save_images_and_update_loss(iteration, scene_name, rendered_image, gt_image, loss.item(), scene.model_path, current_frame_id, save_images=False)
            if iteration > 0 and iteration % 2000 == 0:
                stats_tracker.save_intermediate_report(iteration)
            if iteration in [1000, 5000, 10000, 20000, 30000, 40000]:
                image_debugger.save_specific_iteration_images(iteration, scene_name, current_frame_id, rendered_image, gt_image)
            if iteration in saving_iterations and iteration > warm_up_iter and 'deformed_xyz' in render_pkg:
                print('\n[ITER {}] Saving Gaussians'.format(iteration))
                scene.save_deformed(str(iteration) + '_face', render_pkg['deformed_xyz'], render_pkg['deformed_rotations'], render_pkg['deformed_raw_scale'])
            if iteration in checkpoint_iterations:
                print('\n[ITER {}] Saving Checkpoint'.format(iteration))
                ckpt = (gaussians.capture(), motion_net.state_dict(), motion_optimizer.state_dict(), iteration)
                torch.save(ckpt, scene.model_path + '/chkpnt_face_' + str(iteration) + '.pth')
                torch.save(ckpt, scene.model_path + '/chkpnt_face_latest' + '.pth')
                if clip_color is not None:
                    torch.save({'clips': [dataset.source_path] + [os.path.abspath(e) for e in extra_scenes],
                                'params': clip_color.detach().cpu()}, scene.model_path + '/clip_colors.pth')
                if deform_optimizer is not None:
                    torch.save(gaussians.deform_code.detach().cpu(), scene.model_path + '/deform_code.pth')
                    torch.save(gaussians.deform_decoder.state_dict(), scene.model_path + '/deform_decoder.pth')
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05 + 0.25 * iteration / opt.densify_until_iter, scene.cameras_extent, size_threshold)
                if not mode_long:
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()
            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                from utils.sh_utils import eval_sh
                _n_sh = gaussians.get_features.shape[1]
                shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, _n_sh)
                dir_pp = gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1)
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                _eval_deg = min(gaussians.active_sh_degree, int(round(_n_sh ** 0.5)) - 1)
                sh2rgb = eval_sh(_eval_deg, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                bg_color_mask = (colors_precomp[..., 0] < 30 / 255) * (colors_precomp[..., 1] > 225 / 255) * (colors_precomp[..., 2] < 30 / 255)
                gaussians.prune_points(bg_color_mask.squeeze())
                if not mode_long:
                    gaussians.prune_points((gaussians.get_xyz[:, -1] < -0.07).squeeze())
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                if gaussians.intraoral_rigid and opt.intraoral_contain:
                    from utils import intraoral as _intraoral
                    if iteration % 100 == 0:
                        _own[0] += _intraoral.adopt(gaussians)
                    _own[1] += _intraoral.contain(gaussians)
                    if iteration % 2000 == 0:
                        print(f'\n[intraoral] iter {iteration}: set {int(gaussians.mouth_mask.sum())} of {gaussians._xyz.shape[0]}; '
                              f'since last report adopted {_own[0]}, projections {_own[1]}')
                        _own = [0, 0]
                if iteration > warm_up_iter:
                    motion_optimizer.step()
                    motion_optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    if motion_steps > 0 and iteration - warm_up_iter == motion_steps:
                        for _g in motion_optimizer.param_groups:
                            if _g['name'] in _ARTICULATION_GROUPS:
                                for _p in _g['params']:
                                    _p.requires_grad_(False)
                        print(f'\n[adapt] iter {iteration}: articulation parts of the motion network frozen after {motion_steps} steps')
                    if deform_optimizer is not None:
                        deform_optimizer.step()
                        deform_optimizer.zero_grad(set_to_none=True)
                    if clip_color_opt is not None:
                        if clip_color.grad is not None:
                            clip_color.grad[0].zero_()
                        clip_color_opt.step()
                        clip_color_opt.zero_grad(set_to_none=True)
    print('\n[FINAL] Saving final deformed Gaussians for the first 10 frames of each scene...')
    with torch.no_grad():
        for i, scene_to_save in enumerate(scene_list):
            scene_name = data_list[i]
            train_cameras = scene_to_save.getTrainCameras()
            ply_output_dir = os.path.join(scene_to_save.model_path, 'deformed_plys')
            os.makedirs(ply_output_dir, exist_ok=True)
            if len(train_cameras) > 0:
                num_frames_to_save = min(10, len(train_cameras))
                for frame_idx in range(num_frames_to_save):
                    viewpoint_cam = train_cameras[frame_idx]
                    frame_id = viewpoint_cam.image_name
                    render_pkg = render_motion(viewpoint_cam, scene_to_save.gaussians, motion_net, pipe, background)
                    deformed_xyz = render_pkg.get('deformed_xyz')
                    deformed_rotations = render_pkg.get('deformed_rotations')
                    deformed_raw_scale = render_pkg.get('deformed_raw_scale')
                    if deformed_xyz is not None and deformed_rotations is not None and (deformed_raw_scale is not None):
                        ply_filename = f"{scene_name}_frame_{frame_id}_iter_{iteration}.ply"
                        debug_ply_path = os.path.join(ply_output_dir, ply_filename)
                        scene_to_save.gaussians.save_deformed_ply(deformed_xyz, deformed_rotations, deformed_raw_scale, debug_ply_path)
                    else:
                        print(f"Skipped deformed PLY for scene '{scene_name}', frame '{frame_id}': renderer did not return deformed tensors.")
            else:
                print(f"No train cameras available for scene '{scene_name}'.")
    print('\nOK Finished saving deformed Gaussians.')
    image_debugger.save_final_images()
    stats_tracker.save_final_report(iteration)
    flame_debugger.generate_final_report()
if __name__ == '__main__':
    parser = ArgumentParser(description='Training script parameters')
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--test_iterations', nargs='+', type=int, default=[])
    parser.add_argument('--save_iterations', nargs='+', type=int, default=[])
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--checkpoint_iterations', nargs='+', type=int, default=[])
    parser.add_argument('--start_checkpoint', type=str, default=None)
    parser.add_argument('--long', action='store_true', default=False)
    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--warm_up_iter', type=int, default=500, help='Number of warm-up iterations with static rendering only')
    parser.add_argument('--adapt_adain_only', action='store_true', default=False, help='Freeze the pretrained GRMN and tune only the AdaIN modulation parameters')
    parser.add_argument('--adapt_unfreeze_heads', action='store_true', default=False, help='With --adapt_adain_only, also fine-tune the small FLAME/mouth prediction heads (identity-specific articulation)')
    parser.add_argument('--adapt_unfreeze_backbone', action='store_true', default=False, help='Also fine-tune the last layer of the shared motion MLP')
    parser.add_argument('--au_mode', choices=['decoupled', 'legacy'], default='decoupled', help='decoupled (default): jaw/lower face from audio only, AUs restricted to the upper face. legacy: original additive AU fusion into every head')
    parser.add_argument('--no_window_supervision', action='store_true', default=False, help='Supervise only the center frame of each audio window (original behaviour)')
    parser.add_argument('--w_vel', type=float, default=1.0, help='Weight of the FLAME velocity term (window supervision only)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--motion_steps', type=int, default=0,
                        help='Fit the articulation parts of the motion network (FLAME heads, AdaIN, identity code) for this many '
                             'steps after the warm-up, with the learning rate decayed over these steps, then freeze them '
                             '(0 = for the whole run, the previous behaviour). Recommended: 1000.')
    parser.add_argument('--fixed_background', action='store_true', default=False, help='Train against the fixed green background (original behaviour) instead of a random colour per step')
    parser.add_argument('--extra_scenes', nargs='*', default=[], help='Further processed clips of the same person (other emotions / recordings), each tracked on its own')
    parser.add_argument('--extra_views', type=int, default=None, help='Frames used from each extra clip (default: --N_views)')
    parser.add_argument('--intraoral', choices=['audio', 'rigid'], default='audio',
                        help='audio: intra-oral gaussians with audio-conditioned offsets (original). rigid: teeth / '
                             'mouth-cavity gaussians bound to the teeth geometry of the FLAME mesh, no audio offsets')
    parser.add_argument('--no_audio_lag_calibration', action='store_true', default=False,
                        help='Do not calibrate the audio/motion lag of the prior on the adaptation clip')
    parser.set_defaults(audio_lag=None)
    args = parser.parse_args(sys.argv[1:])
    if args.fixed_background:
        args.random_background = False
    args.save_iterations.append(args.iterations)
    print('Optimizing ' + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.audio_lag is not None:
        print(f'[adapt] using the given --audio_lag {args.audio_lag} (no calibration)')
    elif not args.no_audio_lag_calibration and args.pretrain_path and os.path.exists(args.pretrain_path) and not args.audio_file:
        from utils.audio_lag import calibrate_audio_lag
        _ck = torch.load(args.pretrain_path, map_location='cpu', weights_only=False)
        _state = _ck[0] if isinstance(_ck, (list, tuple)) and len(_ck) == 3 else (_ck[1] if isinstance(_ck, (list, tuple)) else _ck)
        from utils.audio_features import check_prior_extractor
        _state = check_prior_extractor(_state, args.audio_extractor, args.pretrain_path)
        args.audio_lag, _corr = calibrate_audio_lag(os.path.abspath(args.source_path), _state, args.N_views,
                                                    audio_extractor=args.audio_extractor)
        print(f'[adapt] audio lag calibrated on the adaptation clip: {args.audio_lag} frame(s); '
              f'jaw corr by lag: ' + ', '.join(f'{k:+d}:{v:.3f}' for k, v in _corr.items()))
        del _ck, _state
    if args.audio_lag is None:
        args.audio_lag = 0
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.long, args.pretrain_path, args.warm_up_iter, args.adapt_adain_only, args.adapt_unfreeze_heads, args.au_mode, args.adapt_unfreeze_backbone, not args.no_window_supervision, args.w_vel, args.extra_scenes, args.extra_views, args.intraoral, args.motion_steps)
    print('\nTraining complete.')
