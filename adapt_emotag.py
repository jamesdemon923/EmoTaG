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
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, mode_long, pretrain_ckpt_path, warm_up_iter, adapt_adain_only=False, adapt_unfreeze_heads=False):
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
    # Optional override: frequent opacity resets (every warm_up_iter) can suppress
    # fine sharp detail late in adaptation. EMOTAG_OPACITY_RESET sets the interval.
    _opr = os.environ.get('EMOTAG_OPACITY_RESET')
    if _opr:
        opt.opacity_reset_interval = int(_opr)
    opt.densify_until_iter = opt.iterations - 1000
    # Ease the motion deformation in gradually after warm-up (curriculum_weight 0 -> 1)
    # so the full deformation is not applied abruptly to the statically warmed-up
    # Gaussians. Mirrors the pre-training schedule.
    curriculum_transition_iters = max(1, int((opt.iterations - warm_up_iter) * 0.1))
    curriculum_start_iter = warm_up_iter + 1
    curriculum_end_iter = curriculum_start_iter + curriculum_transition_iters
    first_iter = 0
    motion_net = MotionNetwork(args=dataset).cuda()
    motion_optimizer = torch.optim.AdamW(motion_net.get_params(0.005, 0.0005), betas=(0.9, 0.99), eps=1e-08, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(motion_optimizer, lambda iter: 0.1 ** (iter / opt.iterations))
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
    flame_wrapper = SimpleFlameWrapper(flame_params_file=flame_params_file).cuda()
    gaussians = GaussianModel(dataset)
    gaussians.flame_wrapper = flame_wrapper
    # AdaFace identity descriptor.
    identity_path = os.path.join(dataset.source_path, 'identity_feature.npy')
    if os.path.exists(identity_path):
        gaussians.identity_feature = torch.from_numpy(np.load(identity_path).reshape(-1)).float().cuda()
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if pretrain_ckpt_path is None or not os.path.exists(pretrain_ckpt_path):
        sys.exit(
            f'[adapt] --pretrain_path is required and must point to an existing multi-identity '
            f'EMA checkpoint (e.g. output/pretrain/chkpnt_ema_face_latest.pth). Got: {pretrain_ckpt_path}'
        )
    pretrain_ckpt = torch.load(pretrain_ckpt_path, map_location='cpu')
    if isinstance(pretrain_ckpt, (list, tuple)) and len(pretrain_ckpt) == 3:
        # Multi-identity EMA / motion-only checkpoint: (motion_state, optimizer_state, iter).
        motion_params = pretrain_ckpt[0]
    elif isinstance(pretrain_ckpt, (list, tuple)) and len(pretrain_ckpt) == 4:
        # Per-scene checkpoint: (gaussians, motion_state, optimizer_state, iter).
        motion_params = pretrain_ckpt[1]
    elif isinstance(pretrain_ckpt, dict):
        motion_params = pretrain_ckpt
    else:
        sys.exit(
            f'[adapt] Unrecognized pretrain checkpoint structure at {pretrain_ckpt_path}. '
            f'Expected a 3-tuple EMA checkpoint, a 4-tuple per-scene checkpoint, or a raw state_dict.'
        )
    motion_net.load_state_dict(motion_params)
    if adapt_adain_only:
        motion_net.freeze_for_adaptation(unfreeze_heads=adapt_unfreeze_heads)
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
        # Few-shot adaptation: cap the intra-oral subset at a small multiple of its
        # anatomical initialization (see GaussianModel.freeze_mouth_set for the
        # unbounded-growth failure mode; ~1k gaussians are too few for teeth detail).
        gaussians.freeze_mouth_set = True
        gaussians.mouth_budget = int(opt.mouth_budget_factor * max(1, int(mouth_mask.sum().item())))
    except Exception as e:
        sys.exit(f'[adapt] Failed to load mouth_point_indices.npy for this scene: {e}')
    if checkpoint:
        model_params, motion_params, motion_optimizer_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        motion_net.load_state_dict(motion_params)
        motion_optimizer.load_state_dict(motion_optimizer_params)
    # Adaptation caps the SH degree at 1 by default; EMOTAG_ADAPT_SH2=1 keeps the
    # full degree (args.sh_degree) for more view-dependent colour capacity.
    if not mode_long and os.environ.get('EMOTAG_ADAPT_SH2') != '1':
        gaussians.max_sh_degree = 1
    # Expression-conditioned deformation prior (EMOTAG_DEFORM=1): load the shared
    # decoder pretrained across identities (frozen) and fit only a low-dim identity
    # code for this novel subject on the 5s clip. The shared decoder supplies the
    # cross-identity generalization the few-shot data cannot; the tiny code keeps the
    # fit from overfitting the adaptation frames.
    deform_optimizer = None
    if os.environ.get('EMOTAG_DEFORM') == '1':
        from scene.deform_decoder import ExprDeformDecoder
        # EMOTAG_DEFORM_CKPT lets a sweep point at a specific decoder snapshot.
        deform_ckpt = os.environ.get('EMOTAG_DEFORM_CKPT') or \
            os.path.join(os.path.dirname(pretrain_ckpt_path), 'chkpnt_deform_ema_latest.pth')
        if not os.path.exists(deform_ckpt):
            sys.exit(f'[adapt] EMOTAG_DEFORM=1 but no deformation prior at {deform_ckpt}')
        dd = ExprDeformDecoder().cuda()
        dd.load_state_dict(torch.load(deform_ckpt, map_location='cuda:0')[0])
        for p in dd.parameters():
            p.requires_grad_(False)
        dd.eval()
        gaussians.deform_decoder = dd
        # Novel identity: init the code at the pretrained mean and optimize just it.
        init_code = dd.id_codes.detach().mean(dim=0, keepdim=True).clone()
        gaussians.deform_code = torch.nn.Parameter(init_code.cuda())
        deform_optimizer = torch.optim.AdamW([gaussians.deform_code], lr=5e-3, betas=(0.9, 0.99))
        print('loaded expression deformation prior; fitting identity code')
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
        face_mask = torch.as_tensor(unpack_bool_mask(viewpoint_cam.talking_dict['face_mask'])).cuda()
        hair_mask = torch.as_tensor(unpack_bool_mask(viewpoint_cam.talking_dict['hair_mask'])).cuda()
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
            lambda_ssim = opt.lambda_dssim
            photo_l1_loss = l1_loss(rendered_image, gt_image)
            photo_ssim_loss = 1.0 - ssim(rendered_image, gt_image)
            loss = (1.0 - lambda_ssim) * photo_l1_loss + lambda_ssim * photo_ssim_loss
        else:
            # Then optimize the personalized motion response with FLAME supervision.
            if iteration == warm_up_iter + 1:
                print('Starting dynamic motion stage.')
            if iteration < curriculum_start_iter:
                curriculum_weight = 0.0
            elif iteration >= curriculum_end_iter:
                curriculum_weight = 1.0
            else:
                curriculum_weight = min(1.0, max(0.0, (iteration - curriculum_start_iter) / curriculum_transition_iters))
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
            gt_image = gt_image_original * head_mask + background[:, None, None] * ~head_mask
            lambda_ssim = opt.lambda_dssim
            photo_l1_loss = l1_loss(rendered_image, gt_image)
            photo_ssim_loss = 1.0 - ssim(rendered_image, gt_image)
            rendering_loss = (1.0 - lambda_ssim) * photo_l1_loss + lambda_ssim * photo_ssim_loss
            # Perceptual term: keeps high-frequency facial detail that the L1/SSIM
            # optimum blurs away under residual geometric misalignment.
            if opt.w_lpips > 0:
                if not hasattr(training, '_lpips_fn'):
                    training._lpips_fn = lpips.LPIPS(net='alex').eval().cuda()
                    for p in training._lpips_fn.parameters():
                        p.requires_grad_(False)
                lpips_loss = training._lpips_fn(rendered_image.unsqueeze(0), gt_image.unsqueeze(0), normalize=True).squeeze()
                rendering_loss = rendering_loss + opt.w_lpips * lpips_loss
            # Mouth-crop loss: the intra-oral region is visible in only a handful of
            # the adaptation frames and receives almost no gradient share from the
            # full-frame terms; weight the lips_rect crop explicitly (the original
            # pipeline used a dedicated mouth training stage for the same reason).
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
            flame_exp_loss = F.mse_loss(pred_exp_flat, gt_exp_flat)
            flame_jaw_loss = F.mse_loss(pred_jaw_flat, gt_jaw_flat)
            # Jaw is THE lip-sync driver but its MSE (radians, ~1e-3) is drowned by the
            # 50-dim expression MSE (~0.2-0.8); unweighted, the jaw regression tends to
            # collapse toward the mean. Reweight so the jaw term matches the
            # rendering-loss scale.
            flame_reg_loss = flame_exp_loss + opt.w_jaw_reg * flame_jaw_loss
            w_rendering = 1.0
            w_flame_reg = 0.1
            loss = w_rendering * rendering_loss + w_flame_reg * flame_reg_loss
            # Semantic Emotion Guidance (KL + score distillation).
            emo_target = viewpoint_cam.talking_dict.get('emotion', None)
            if emo_target is not None:
                seg_loss, _, _ = semantic_emotion_guidance_loss(render_pkg.get('emotion_logits'), render_pkg.get('gate'), emo_target, viewpoint_cam.talking_dict.get('emotion_score', 0.0), w_kl=opt.w_emotion_kl, w_score=opt.w_emotion_score)
                loss = loss + seg_loss
            # Geometric loss against Sapiens depth/normal pseudo ground-truth.
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
        # GaussianAvatars anti-drift regularizers, in triangle-local units: keep
        # gaussians within ~1 triangle of their parent and no larger than ~0.6
        # triangle sizes. Without these, offsets grow until the avatar only
        # renders correctly on the adaptation frames. In 'local' binding mode the
        # raw parameters are already local; in 'hybrid' mode _xyz is a world
        # offset, so normalize by the parent triangle size first.
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
                loss = loss + opt.lambda_xyz * F.relu(local_off[vis] - opt.threshold_xyz).mean()
            if opt.lambda_scale > 0:
                loss = loss + opt.lambda_scale * F.relu(local_scale[vis] - opt.threshold_scale).norm(dim=1).mean()
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
            if iteration in saving_iterations and iteration > warm_up_iter and 'deformed_xyz' in render_pkg:
                print('\n[ITER {}] Saving Gaussians'.format(iteration))
                scene.save_deformed(str(iteration) + '_face', render_pkg['deformed_xyz'], render_pkg['deformed_rotations'], render_pkg['deformed_raw_scale'])
            if iteration in checkpoint_iterations:
                print('\n[ITER {}] Saving Checkpoint'.format(iteration))
                ckpt = (gaussians.capture(), motion_net.state_dict(), motion_optimizer.state_dict(), iteration)
                torch.save(ckpt, scene.model_path + '/chkpnt_face_' + str(iteration) + '.pth')
                torch.save(ckpt, scene.model_path + '/chkpnt_face_latest' + '.pth')
                if deform_optimizer is not None:
                    torch.save(gaussians.deform_code.detach().cpu(), scene.model_path + '/deform_code.pth')
                    # Self-contained: stash the frozen shared decoder next to the code so
                    # evaluation needs only the model dir.
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
                # Reshape by the ACTUAL SH coeff count (get_features dim 1) rather than
                # (max_sh_degree+1)**2: in adaptation those can disagree (features were
                # allocated at a higher degree than the max_sh_degree attribute reports),
                # which otherwise makes this view() raise a shape error.
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
                if iteration > warm_up_iter:
                    motion_optimizer.step()
                    motion_optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    if deform_optimizer is not None:
                        deform_optimizer.step()
                        deform_optimizer.zero_grad(set_to_none=True)
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
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print('Optimizing ' + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.long, args.pretrain_path, args.warm_up_iter, args.adapt_adain_only, args.adapt_unfreeze_heads)
    print('\nTraining complete.')
