import torch
import math
import torch.nn.functional as F
import sys
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.motion_net import MotionNetwork
from utils.sh_utils import eval_sh
import numpy as np
def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    canonical_xyz = pc.get_canonical_xyz
    screenspace_points = torch.zeros_like(canonical_xyz, dtype=canonical_xyz.dtype, requires_grad=True, device='cuda') + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(image_height=int(viewpoint_camera.image_height), image_width=int(viewpoint_camera.image_width), tanfovx=tanfovx, tanfovy=tanfovy, bg=bg_color, scale_modifier=scaling_modifier, viewmatrix=viewpoint_camera.world_view_transform, projmatrix=viewpoint_camera.full_proj_transform, sh_degree=pc.active_sh_degree, campos=viewpoint_camera.camera_center, prefiltered=False, debug=pipe.debug)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = canonical_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, extra = rasterizer(means3D=means3D, means2D=means2D, shs=shs, colors_precomp=colors_precomp, opacities=opacity, scales=scales, rotations=rotations, cov3Ds_precomp=cov3D_precomp, extra_attrs=torch.ones_like(opacity))
    return {'render': rendered_image, 'viewspace_points': screenspace_points, 'visibility_filter': radii > 0, 'depth': rendered_depth, 'alpha': rendered_alpha, 'normal': rendered_norm, 'radii': radii}

def render_motion(viewpoint_camera, pc: GaussianModel, motion_net: MotionNetwork, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, debug_viz=True, curriculum_weight=1.0, drive_with_gt=False, drive_params=None, mouth_params=None, return_motion_only=False):
    """
    Render the scene using the new FLAME-driven architecture.
    """
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(image_height=int(viewpoint_camera.image_height), image_width=int(viewpoint_camera.image_width), tanfovx=tanfovx, tanfovy=tanfovy, bg=bg_color, scale_modifier=scaling_modifier, viewmatrix=viewpoint_camera.world_view_transform, projmatrix=viewpoint_camera.full_proj_transform, sh_degree=pc.active_sh_degree, campos=viewpoint_camera.camera_center, prefiltered=False, debug=pipe.debug)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    audio_feat = viewpoint_camera.talking_dict['auds'].cuda()
    au_feat = viewpoint_camera.talking_dict['au_features'].cuda()
    identity_feat = getattr(pc, 'identity_feature', None)
    if identity_feat is not None:
        identity_feat = identity_feat.cuda()
    # Per-identity learnable code: gives each pretrain identity dedicated capacity so
    # the shared motion network is not forced to average across all of them.
    ind_code = None
    if getattr(motion_net, 'individual_dim', 0) > 0:
        _idx = int(getattr(pc, 'identity_index', 0))
        ind_code = motion_net.individual_codes[_idx:_idx + 1]
    preds = motion_net(audio_feat, c=ind_code, au=au_feat, identity=identity_feat)
    pred_exp_raw = preds['flame_exp']
    pred_jaw_raw = preds['flame_jaw']
    if pred_exp_raw.dim() == 1:
        pred_exp_raw = pred_exp_raw.unsqueeze(0)
    if pred_jaw_raw.dim() == 1:
        pred_jaw_raw = pred_jaw_raw.unsqueeze(0)
    # The audio window holds per-frame predictions for frames [t-4, t+4); use the
    # center frame's prediction rather than averaging, which would low-pass the
    # articulation timing. The encoder still attends over the whole window as context.
    if pred_exp_raw.shape[0] > 1:
        pred_exp_raw = pred_exp_raw[pred_exp_raw.shape[0] // 2].unsqueeze(0)
    if pred_jaw_raw.shape[0] > 1:
        pred_jaw_raw = pred_jaw_raw[pred_jaw_raw.shape[0] // 2].unsqueeze(0)
    encoded_audio_feat = preds.get('encoded_audio', None)
    pred_exp = pred_exp_raw * curriculum_weight
    pred_jaw = pred_jaw_raw * curriculum_weight
    frame_id = viewpoint_camera.talking_dict['frame_id']
    flame_params_gt = pc.flame_wrapper.load_flame_params_from_npz(frame_id)
    gt_exp_feat = flame_params_gt['expression']
    gt_pose = flame_params_gt['pose']
    if gt_exp_feat.dim() == 1:
        gt_exp_feat = gt_exp_feat.unsqueeze(0)
    if gt_pose.dim() == 1:
        gt_pose = gt_pose.unsqueeze(0)
    gt_global_rot = gt_pose[:, :3]
    if drive_with_gt:
        # Appearance optimization uses the tracked FLAME parameters so the deformed
        # geometry is aligned with the target frame; the motion network still trains
        # through the parameter-regression / SEG losses computed from `preds` above.
        # Driving the photometric loss with the (imperfect) predicted parameters
        # supervises the gaussians through misaligned geometry every step, and the
        # optimum of that objective is a motion-blurred average texture.
        drive_exp = gt_exp_feat[:, :50]
        drive_jaw = gt_pose[:, 3:]
    elif drive_params is not None:
        # Externally supplied (e.g. temporally smoothed) expression/jaw: inference
        # smooths the per-frame predictions over time to suppress frame-to-frame
        # parameter tremor without touching articulation timing.
        drive_exp, drive_jaw = drive_params
        if drive_exp.dim() == 1:
            drive_exp = drive_exp.unsqueeze(0)
        if drive_jaw.dim() == 1:
            drive_jaw = drive_jaw.unsqueeze(0)
    else:
        drive_exp = pred_exp
        drive_jaw = pred_jaw
    final_pose = torch.cat([gt_global_rot, drive_jaw], dim=1)
    flame_verts = pc.flame_wrapper.get_flame_vertices(frame_id, motion_expression=drive_exp, motion_jaw_pose=drive_jaw)
    canonical_xyz = pc.get_canonical_xyz
    canonical_scales = pc.get_scaling
    canonical_rotations = pc.get_rotation
    deformed_gaussians = pc.get_deformed_gaussians(flame_verts, final_pose)
    deformed_xyz = deformed_gaussians['xyz']
    deformed_scales = deformed_gaussians['scale']
    deformed_rotations = deformed_gaussians['rotation']
    w = curriculum_weight
    means3D = (1.0 - w) * canonical_xyz + w * deformed_xyz
    scales = (1.0 - w) * canonical_scales + w * deformed_scales
    rotations = (1.0 - w) * canonical_rotations + w * deformed_rotations
    rotations = F.normalize(rotations, p=2, dim=1)
    opacity = pc.get_opacity
    cov3D_precomp = None
    colors_precomp = None
    shs = pc.get_features
    mouth_pred = None
    mouth_mask = getattr(pc, 'mouth_mask', None)
    if mouth_mask is not None and mouth_mask.numel() == means3D.shape[0] and mouth_mask.any():
        if mouth_params is not None:
            # Externally supplied (e.g. temporally smoothed) intra-oral deltas: the
            # triplane branch predicts them per frame from audio, so they carry the
            # same frame-to-frame tremor as the expression head.
            mouth_pred = mouth_params
        else:
            mouth_pred = motion_net.forward_mouth(xyz=means3D, a=audio_feat, mouth_mask=mouth_mask, c=ind_code, encoded_audio=encoded_audio_feat)
        if return_motion_only:
            return {'predicted_flame_exp': pred_exp, 'predicted_flame_jaw': pred_jaw,
                    'mouth_pred': ({k: v.detach() for k, v in mouth_pred.items()} if mouth_pred is not None else None)}
        if mouth_pred is not None and mouth_pred['d_mu'].numel() > 0:
            means3D = means3D.clone()
            rotations = rotations.clone()
            scales = scales.clone()
            weighted_d_mu = mouth_pred['d_mu'] * curriculum_weight
            weighted_d_rot = mouth_pred['d_rot'] * curriculum_weight
            weighted_d_scale = mouth_pred['d_scale'] * curriculum_weight
            means3D[mouth_mask] = means3D[mouth_mask] + weighted_d_mu
            mouth_rotations_current = rotations[mouth_mask]
            mouth_delta_rotations = pc.rotation_activation(weighted_d_rot)
            q1 = mouth_rotations_current
            q2 = mouth_delta_rotations
            w1, x1, y1, z1 = (q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3])
            w2, x2, y2, z2 = (q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3])
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            composed_rotations = torch.stack([w, x, y, z], dim=1)
            composed_rotations = pc.rotation_activation(composed_rotations)
            rotations[mouth_mask] = composed_rotations
            mouth_scale_delta = pc.scaling_activation(weighted_d_scale)
            scales[mouth_mask] = scales[mouth_mask] * mouth_scale_delta
    if return_motion_only:
        # Scene without an intra-oral subset: nothing further to collect.
        return {'predicted_flame_exp': pred_exp, 'predicted_flame_jaw': pred_jaw, 'mouth_pred': None}
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device='cuda') + 0
    if screenspace_points.requires_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass
    means2D = screenspace_points
    rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, extra = rasterizer(means3D=means3D, means2D=means2D, shs=shs, colors_precomp=colors_precomp, opacities=opacity, scales=scales, rotations=rotations, cov3Ds_precomp=None)
    return_dict = {'render': rendered_image, 'viewspace_points': screenspace_points, 'visibility_filter': radii > 0, 'depth': rendered_depth, 'alpha': rendered_alpha, 'normal': rendered_norm, 'radii': radii, 'predicted_flame_exp': pred_exp_raw, 'predicted_flame_jaw': pred_jaw_raw, 'gt_flame_exp': gt_exp_feat, 'gt_flame_pose': gt_pose, 'deformed_flame_verts': flame_verts, 'deformed_xyz': means3D, 'deformed_rotations': rotations, 'deformed_raw_scale': deformed_gaussians['raw_scale'], 'gate': preds.get('gate'), 'emotion_logits': preds.get('emotion_logits')}
    return return_dict
