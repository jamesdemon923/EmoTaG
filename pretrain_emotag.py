import os
import random
_mpl_config_dir = os.environ.get('MPLCONFIGDIR')
if not _mpl_config_dir or not os.access(_mpl_config_dir, os.W_OK):
    os.environ['MPLCONFIGDIR'] = os.path.join('/tmp', 'emotag_mpl')
import torch
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage
from random import randint
from utils.loss_utils import l1_loss, l2_loss, patchify, ssim, semantic_emotion_guidance_loss
from gaussian_renderer import render, render_motion
import sys, copy
from scene_pretrain import Scene, GaussianModel, MotionNetwork
from utils.general_utils import safe_state
import lpips
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.flame_wrapper import SimpleFlameWrapper
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import cv2
import json
from utils.visualize_utils import ImageDebugger, TrainingStatsTracker, FlameDebugger

def _split_scene_names(raw_scene_names):
    names = []
    for chunk in raw_scene_names.replace('\n', ',').split(','):
        name = chunk.strip()
        if name:
            names.append(name)
    return names

def _read_scene_names_file(path):
    names = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            names.extend(_split_scene_names(line))
    return names

def _has_processed_scene_files(path):
    required = ['transforms.json', 'flame_params.npz', 'model_center.npy', 'mouth_point_indices.npy', 'points3D.ply', 'face_indices.npy', 'bary_coords.npy']
    return all((os.path.exists(os.path.join(path, name)) for name in required))

def _discover_processed_scenes(source_path, max_depth):
    source_path = os.path.abspath(source_path)
    if _has_processed_scene_files(source_path):
        return ['.']
    discovered = []
    for dirpath, dirnames, _filenames in os.walk(source_path):
        current_depth = len(os.path.relpath(dirpath, source_path).split(os.sep))
        if dirpath == source_path:
            current_depth = 0
        if current_depth > max_depth:
            dirnames[:] = []
            continue
        if _has_processed_scene_files(dirpath):
            discovered.append(os.path.relpath(dirpath, source_path))
            dirnames[:] = []
    return sorted(discovered)

def resolve_pretrain_scene_names(dataset):
    data_list = []
    if getattr(dataset, 'scene_names_file', ''):
        scene_names_file = os.path.abspath(dataset.scene_names_file)
        if not os.path.exists(scene_names_file):
            raise FileNotFoundError(f'Scene names file not found: {scene_names_file}')
        data_list.extend(_read_scene_names_file(scene_names_file))
    if getattr(dataset, 'scene_names', ''):
        data_list.extend(_split_scene_names(dataset.scene_names))
    if not data_list:
        data_list = _discover_processed_scenes(dataset.source_path, max_depth=getattr(dataset, 'scene_discovery_depth', 4))
    normalized = []
    seen = set()
    for name in data_list:
        if os.path.isabs(name):
            rel_name = os.path.relpath(name, dataset.source_path)
        else:
            rel_name = name
        rel_name = os.path.normpath(rel_name)
        if rel_name.startswith('..'):
            raise ValueError(f'Scene path escapes the dataset root: {name}')
        if rel_name not in seen:
            normalized.append(rel_name)
            seen.add(rel_name)
    if not normalized:
        raise RuntimeError('No pretrain scenes found. Pass --scene_names, --scene_names_file, or preprocess data so scene directories contain transforms.json, flame_params.npz, and points3D.ply.')
    invalid = [name for name in normalized if not _has_processed_scene_files(os.path.join(dataset.source_path, name))]
    if invalid:
        joined = ', '.join(invalid)
        raise RuntimeError(f'Pretrain scene(s) are not fully processed: {joined}')
    print('OK Pretrain scenes:')
    for name in normalized:
        print(f'  - {name}')
    return normalized

def training(dataset, opt, pipe, saving_iterations, checkpoint_iterations, checkpoint, debug_from, share_audio_net):
    data_list = resolve_pretrain_scene_names(dataset)
    warm_up_iter = opt.warm_up_iter_per_scene * len(data_list)
    new_densify_from_iter = warm_up_iter + 100
    opt.densify_from_iter = new_densify_from_iter
    opt.densify_until_iter = opt.iterations * len(data_list) * 0.85
    checkpoint_iterations = saving_iterations = [i * len(data_list) for i in range(0, opt.iterations + 1, 10000)] + [opt.iterations * len(data_list)]
    opt.iterations *= len(data_list)
    transition_ratio = 0.1
    curriculum_transition_iters = int(opt.iterations * transition_ratio)
    curriculum_start_iter = warm_up_iter + 1
    curriculum_end_iter = curriculum_start_iter + curriculum_transition_iters
    first_iter = 0
    motion_net = MotionNetwork(args=dataset).cuda()
    motion_optimizer = torch.optim.AdamW(motion_net.get_params(0.005, 5e-05), betas=(0.9, 0.99), eps=1e-08)
    scheduler = torch.optim.lr_scheduler.LambdaLR(motion_optimizer, lambda iter: 0.1 ** (iter / opt.iterations))
    ema_motion_net = ExponentialMovingAverage(motion_net.parameters(), decay=0.995)
    scene_list = []
    scene_paths = {}
    for data_name in data_list:
        _dataset = copy.deepcopy(dataset)
        _dataset.source_path = os.path.join(dataset.source_path, data_name)
        _dataset.model_path = os.path.join(dataset.model_path, data_name)
        scene_paths[data_name] = _dataset.model_path
        os.makedirs(_dataset.model_path, exist_ok=True)
        with open(os.path.join(_dataset.model_path, 'cfg_args'), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(_dataset))))
        flame_params_file = os.path.join(_dataset.source_path, 'flame_params.npz')
        if not os.path.exists(flame_params_file):
            sys.exit(1)
        model_center_path = os.path.join(_dataset.source_path, 'model_center.npy')
        if not os.path.exists(model_center_path):
            sys.exit(1)
        model_center = np.load(model_center_path)
        flame_wrapper = SimpleFlameWrapper(flame_params_file=flame_params_file).cuda()
        gaussians = GaussianModel(dataset)
        gaussians.flame_wrapper = flame_wrapper
        # AdaFace identity descriptor.
        identity_path = os.path.join(_dataset.source_path, 'identity_feature.npy')
        if os.path.exists(identity_path):
            gaussians.identity_feature = torch.from_numpy(np.load(identity_path).reshape(-1)).float().cuda()
        _dataset.model_center = model_center
        scene = Scene(_dataset, gaussians, debug_pre_binding=True)
        scene_list.append(scene)
        gaussians.training_setup(opt)
        try:
            mouth_idx_path = os.path.join(_dataset.source_path, 'mouth_point_indices.npy')
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
        except Exception as e:
            sys.exit(1)
    bg_color = [0, 1, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), ascii=True, dynamic_ncols=True, desc='Training progress')
    first_iter += 1
    image_debugger = ImageDebugger(dataset.model_path, data_list, warm_up_iter, scene_paths)
    stats_tracker = TrainingStatsTracker(data_list, dataset.model_path)
    flame_debugger = FlameDebugger(dataset.model_path, debug_save_interval=10)
    stats_tracker.initialize_from_scenes(scene_list)
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        cur_scene_idx = randint(0, len(scene_list) - 1)
        scene = scene_list[cur_scene_idx]
        gaussians = scene.gaussians
        current_scene_name = data_list[cur_scene_idx]
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        train_cameras = scene.getTrainCameras()
        viewpoint_cam = train_cameras[randint(0, len(train_cameras) - 1)]
        current_frame_id = viewpoint_cam.image_name
        stats_tracker.record_training(current_scene_name, current_frame_id)
        if iteration - 1 == debug_from:
            pipe.debug = True
        face_mask = torch.as_tensor(viewpoint_cam.talking_dict['face_mask']).cuda()
        hair_mask = torch.as_tensor(viewpoint_cam.talking_dict['hair_mask']).cuda()
        head_mask = face_mask
        if iteration <= warm_up_iter:
            # Warm up the Gaussian avatar before enabling audio-driven motion.
            if iteration == 1:
                print('Starting static warm-up stage.')
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            rendered_image = render_pkg['render']
            alpha = render_pkg['alpha']
            viewspace_point_tensor = render_pkg['viewspace_points']
            visibility_filter = render_pkg['visibility_filter']
            radii = render_pkg['radii']
            gt_image_original = viewpoint_cam.original_image.cuda() / 255.0
            gt_image = gt_image_original * head_mask + background[:, None, None] * ~head_mask
            lambda_ssim = 0.2
            photo_l1_loss = l1_loss(rendered_image, gt_image)
            photo_ssim_loss = 1.0 - ssim(rendered_image, gt_image)
            loss = (1.0 - lambda_ssim) * photo_l1_loss + lambda_ssim * photo_ssim_loss
            loss.backward()
        else:
            # Blend from static rendering to the full motion network with a short curriculum.
            if iteration == warm_up_iter + 1:
                print('Starting dynamic motion stage.')
            if iteration < curriculum_start_iter:
                curriculum_weight = 0.0
            elif iteration >= curriculum_end_iter:
                curriculum_weight = 1.0
            else:
                progress = (iteration - curriculum_start_iter) / curriculum_transition_iters
                curriculum_weight = min(1.0, max(0.0, progress))
            if iteration % 100 == 0 or iteration == curriculum_start_iter:
                print(f'Curriculum weight: {curriculum_weight:.4f}')
            render_pkg = render_motion(viewpoint_cam, gaussians, motion_net, pipe, background, curriculum_weight=curriculum_weight)
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
            gt_image = gt_image_original * head_mask + background[:, None, None] * ~head_mask
            lambda_ssim = 0.2
            photo_l1_loss = l1_loss(rendered_image, gt_image)
            photo_ssim_loss = 1.0 - ssim(rendered_image, gt_image)
            rendering_loss = (1.0 - lambda_ssim) * photo_l1_loss + lambda_ssim * photo_ssim_loss
            flame_exp_loss = F.mse_loss(pred_exp_flat, gt_exp_flat)
            flame_jaw_loss = F.mse_loss(pred_jaw_flat, gt_jaw_flat)
            flame_reg_loss = flame_exp_loss + flame_jaw_loss
            w_rendering = 1.0
            w_flame_reg = opt.w_flame_reg
            loss = w_rendering * rendering_loss + w_flame_reg * flame_reg_loss
            # Semantic Emotion Guidance (KL + score distillation).
            emo_target = viewpoint_cam.talking_dict.get('emotion', None)
            if emo_target is not None:
                seg_loss, _, _ = semantic_emotion_guidance_loss(render_pkg.get('emotion_logits'), render_pkg.get('gate'), emo_target, viewpoint_cam.talking_dict.get('emotion_score', 0.0), w_kl=opt.w_emotion_kl, w_score=opt.w_emotion_score)
                loss = loss + seg_loss
            loss.backward()
        if iteration > warm_up_iter:
            flame_debugger.record_flame_params(iteration, current_scene_name, current_frame_id, pred_exp, pred_jaw, gt_exp, gt_jaw, rendering_loss, flame_reg_loss)
        if iteration == 1:
            image_debugger.save_debug_comparison(gt_image_original, head_mask, gt_image, background)
        image_debugger.save_images_and_update_loss(iteration, current_scene_name, rendered_image, gt_image, loss.item(), scene.model_path, current_frame_id, save_images=False)
        if iteration % 2000 == 0:
            stats_tracker.save_intermediate_report(iteration)
        if iteration in [1000, 2000, 5000, 10000, 20000, 30000, 40000]:
            image_debugger.save_specific_iteration_images(iteration, current_scene_name, current_frame_id, rendered_image, gt_image)
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
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05 + 0.25 * iteration / opt.densify_until_iter, scene.cameras_extent, size_threshold)
            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                from utils.sh_utils import eval_sh
                shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
                dir_pp = gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1)
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                bg_color_mask = (colors_precomp[..., 0] < 30 / 255) * (colors_precomp[..., 1] > 225 / 255) * (colors_precomp[..., 2] < 30 / 255)
                gaussians.prune_points(bg_color_mask.squeeze())
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                if iteration > warm_up_iter:
                    motion_optimizer.step()
                    motion_optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    ema_motion_net.update()
            if iteration in checkpoint_iterations:
                print('\n[ITER {}] Saving Checkpoint'.format(iteration))
                ckpt = (motion_net.state_dict(), motion_optimizer.state_dict(), iteration)
                torch.save(ckpt, dataset.model_path + '/chkpnt_face_latest' + '.pth')
                with ema_motion_net.average_parameters():
                    ckpt_ema = (motion_net.state_dict(), motion_optimizer.state_dict(), iteration)
                    torch.save(ckpt, dataset.model_path + '/chkpnt_ema_face_latest' + '.pth')
                for _scene in scene_list:
                    _gaussians = _scene.gaussians
                    ckpt = (_gaussians.capture(), motion_net.state_dict(), motion_optimizer.state_dict(), iteration)
                    torch.save(ckpt, _scene.model_path + '/chkpnt_face_' + str(iteration) + '.pth')
                    torch.save(ckpt, _scene.model_path + '/chkpnt_face_latest' + '.pth')
    image_debugger.save_final_images()
    print('\n[FINAL] Saving final deformed Gaussians for the first 10 frames of each scene...')
    with torch.no_grad():
        # Save compact geometry snapshots for quick inspection.
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
    print('\n[FINAL] Saving Final Checkpoint')
    ckpt = (motion_net.state_dict(), motion_optimizer.state_dict(), iteration)
    torch.save(ckpt, dataset.model_path + '/chkpnt_face_latest' + '.pth')
    with ema_motion_net.average_parameters():
        ckpt_ema = (motion_net.state_dict(), motion_optimizer.state_dict(), iteration)
        torch.save(ckpt, dataset.model_path + '/chkpnt_ema_face_latest' + '.pth')
    for _scene in scene_list:
        _gaussians = _scene.gaussians
        ckpt = (_gaussians.capture(), motion_net.state_dict(), motion_optimizer.state_dict(), iteration)
        torch.save(ckpt, _scene.model_path + '/chkpnt_face_' + str(iteration) + '.pth')
        torch.save(ckpt, _scene.model_path + '/chkpnt_face_latest' + '.pth')
    print('Final checkpoint saved successfully!')
    stats_tracker.save_final_report(iteration)
    flame_debugger.generate_final_summary()
if __name__ == '__main__':
    parser = ArgumentParser(description='Training script parameters')
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--save_iterations', nargs='+', type=int, default=[])
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--checkpoint_iterations', nargs='+', type=int, default=[])
    parser.add_argument('--start_checkpoint', type=str, default=None)
    parser.add_argument('--share_audio_net', action='store_true', default=False)
    parser.add_argument('--dry_run', action='store_true', help='Validate pretrain scene selection without CUDA/training')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print('Optimizing ' + args.model_path)
    if args.dry_run:
        dataset_args = lp.extract(args)
        scene_names = resolve_pretrain_scene_names(dataset_args)
        sys.exit(0)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.share_audio_net)
    print('\nTraining complete.')
