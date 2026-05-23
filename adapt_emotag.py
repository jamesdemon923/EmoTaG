import os
import random
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, l2_loss, patchify, ssim, normalize
from gaussian_renderer import render, render_motion
import sys
from scene import Scene, GaussianModel, MotionNetwork
from utils.general_utils import safe_state
import lpips
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.normal_utils import depth_to_normal
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.flame_wrapper import SimpleFlameWrapper
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
import pdb

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, mode_long, pretrain_ckpt_path, warm_up_iter):
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
    opt.densify_until_iter = opt.iterations - 1000
    first_iter = 0
    motion_net = MotionNetwork(args=dataset).cuda()
    motion_optimizer = torch.optim.AdamW(motion_net.get_params(0.005, 0.0005), betas=(0.9, 0.99), eps=1e-08, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(motion_optimizer, lambda iter: 0.1 ** (iter / opt.iterations))
    os.makedirs(dataset.model_path, exist_ok=True)
    flame_params_file = os.path.join(dataset.source_path, 'flame_params.npz')
    if not os.path.exists(flame_params_file):
        sys.exit(1)
    model_center_path = os.path.join(dataset.source_path, 'model_center.npy')
    if not os.path.exists(model_center_path):
        sys.exit(1)
    model_center = np.load(model_center_path)
    flame_wrapper = SimpleFlameWrapper(flame_params_file=flame_params_file).cuda()
    gaussians = GaussianModel(dataset)
    gaussians.flame_wrapper = flame_wrapper
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    motion_params, _, _ = torch.load(pretrain_ckpt_path)
    motion_net.load_state_dict(motion_params)
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
    except Exception as e:
        sys.exit(1)
    if checkpoint:
        model_params, motion_params, motion_optimizer_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        motion_net.load_state_dict(motion_params)
        motion_optimizer.load_state_dict(motion_optimizer_params)
    if not mode_long:
        gaussians.max_sh_degree = 1
    bg_color = [0, 1, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), ascii=True, dynamic_ncols=True, desc='Training progress')
    first_iter += 1
    data_list = [scene_name]
    scene_paths = {scene_name: dataset.model_path}
    image_debugger = ImageDebugger(dataset.model_path, data_list, warm_up_iter, scene_paths)
    stats_tracker = TrainingStatsTracker(data_list, dataset.model_path)
    flame_debugger = FlameDebugger(dataset.model_path, debug_save_interval=10)
    scene_list = [scene]
    stats_tracker.initialize_from_scenes(scene_list)
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        current_frame_id = viewpoint_cam.image_name
        stats_tracker.record_training(scene_name, current_frame_id)
        if iteration - 1 == debug_from:
            pipe.debug = True
        face_mask = torch.as_tensor(viewpoint_cam.talking_dict['face_mask']).cuda()
        hair_mask = torch.as_tensor(viewpoint_cam.talking_dict['hair_mask']).cuda()
        head_mask = face_mask
        if iteration <= warm_up_iter:
            # First stabilize the identity-specific Gaussian appearance.
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
        else:
            # Then optimize the personalized motion response with FLAME supervision.
            if iteration == warm_up_iter + 1:
                print('Starting dynamic motion stage.')
            render_pkg = render_motion(viewpoint_cam, gaussians, motion_net, pipe, background, debug_viz=iteration == 1)
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
            w_flame_reg = 0.1
            loss = w_rendering * rendering_loss + w_flame_reg * flame_reg_loss
            flame_debugger.record_flame_params(iteration, scene_name, current_frame_id, pred_exp, pred_jaw, gt_exp, gt_jaw, rendering_loss, flame_reg_loss)
        loss.backward()
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
            if iteration in saving_iterations:
                print('\n[ITER {}] Saving Gaussians'.format(iteration))
                scene.save_deformed(str(iteration) + '_face', deformed_xyz, deformed_rotations)
            if iteration in checkpoint_iterations:
                print('\n[ITER {}] Saving Checkpoint'.format(iteration))
                ckpt = (gaussians.capture(), motion_net.state_dict(), motion_optimizer.state_dict(), iteration)
                torch.save(ckpt, scene.model_path + '/chkpnt_face_' + str(iteration) + '.pth')
                torch.save(ckpt, scene.model_path + '/chkpnt_face_latest' + '.pth')
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
                shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
                dir_pp = gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1)
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                bg_color_mask = (colors_precomp[..., 0] < 30 / 255) * (colors_precomp[..., 1] > 225 / 255) * (colors_precomp[..., 2] < 30 / 255)
                gaussians.prune_points(bg_color_mask.squeeze())
                if not mode_long:
                    gaussians.prune_points((gaussians.get_xyz[:, -1] < -0.07).squeeze())
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                if iteration > warm_up_iter:
                    motion_optimizer.step()
                    motion_optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
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
    image_debugger.save_final_images()
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
    parser.add_argument('--test_iterations', nargs='+', type=int, default=[])
    parser.add_argument('--save_iterations', nargs='+', type=int, default=[])
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--checkpoint_iterations', nargs='+', type=int, default=[])
    parser.add_argument('--start_checkpoint', type=str, default=None)
    parser.add_argument('--long', action='store_true', default=False)
    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--warm_up_iter', type=int, default=500, help='Number of warm-up iterations with static rendering only')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print('Optimizing ' + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.long, args.pretrain_path, args.warm_up_iter)
    print('\nTraining complete.')
