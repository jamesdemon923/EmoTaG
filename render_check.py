"""Quick correctness render: load a pretrain scene's checkpoint with the CORRECT
per-identity index and render one test-split frame next to its GT."""
import copy, os, sys
import numpy as np
import torch
from PIL import Image
from arguments import ModelParams, PipelineParams
from argparse import ArgumentParser
from gaussian_renderer import render_motion
from scene import GaussianModel, MotionNetwork, Scene
from scene.flame_wrapper import SimpleFlameWrapper
from utils.general_utils import safe_state


def build_index(list_file):
    names = []
    for ln in open(list_file):
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        names.append(s)
    return names


def main():
    p = ArgumentParser()
    lp = ModelParams(p, sentinel=True)
    pp = PipelineParams(p)
    p.add_argument('--scene', required=True)
    p.add_argument('--list_file', default='data/hdtf_pretrain_clean68.txt')
    p.add_argument('--frame', type=int, default=80)
    p.add_argument('--ckpt', default='chkpnt_face_latest.pth')
    p.add_argument('--model_root', default='output/pretrain_ddp_long')
    p.add_argument('--cw', type=float, default=1.0)
    p.add_argument('--out', required=True)
    args = p.parse_args(sys.argv[1:])
    safe_state(False)

    full = build_index(args.list_file)
    idx = full.index(args.scene)
    print(f'scene {args.scene} -> identity_index {idx}')

    dataset = lp.extract(args)
    dataset.source_path = os.path.join('data', args.scene)
    dataset.model_path = os.path.join(args.model_root, args.scene)
    dataset.type = 'face'
    dataset.preload = True
    if getattr(dataset, 'sh_degree', None) is None:
        dataset.sh_degree = 2
    if getattr(dataset, 'white_background', None) is None:
        dataset.white_background = False
    if not getattr(dataset, 'audio_file', None):
        dataset.audio_file = os.path.join(dataset.source_path, 'aud_w2v.npy')

    with torch.no_grad():
        gaussians = GaussianModel(copy.deepcopy(dataset))
        flame_params_file = os.path.join(dataset.source_path, 'flame_params.npz')
        dataset.model_center = np.load(os.path.join(dataset.source_path, 'model_center.npy'))
        gaussians.flame_wrapper = SimpleFlameWrapper(flame_params_file=flame_params_file).cuda()
        idp = os.path.join(dataset.source_path, 'identity_feature.npy')
        if os.path.exists(idp):
            gaussians.identity_feature = torch.from_numpy(np.load(idp).reshape(-1)).float().cuda()
        gaussians.identity_index = idx
        scene = Scene(dataset, gaussians, shuffle=False)
        motion_net = MotionNetwork(args=dataset).cuda()
        ckpt = os.path.join(dataset.model_path, args.ckpt)
        model_params, motion_params, _, _ = torch.load(ckpt, map_location='cuda:0')
        motion_net.load_state_dict(motion_params, strict=False)
        gaussians.restore(model_params, None)
        # mouth mask (so mouth deformation path runs like training)
        try:
            mi = np.load(os.path.join(dataset.source_path, 'mouth_point_indices.npy')).reshape(-1)
            N = gaussians.get_xyz.shape[0]
            mit = torch.from_numpy(mi).long().clamp_(0, max(0, N - 1))
            mm = torch.zeros(N, dtype=torch.bool, device='cuda')
            mm[mit.cuda()] = True
            gaussians.mouth_mask = mm
            gaussians.mouth_indices = mit.cuda()
        except Exception as e:
            print('mouth setup skipped:', e)

        bg = torch.tensor([0, 1, 0], dtype=torch.float32, device='cuda')
        views = scene.getTestCameras()
        if not views:
            views = scene.getTrainCameras()
        view = views[min(args.frame, len(views) - 1)]
        pkg = render_motion(view, gaussians, motion_net, pp.extract(args), bg, curriculum_weight=args.cw)
        img = pkg['render'].clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)
        gt = (view.original_image.cuda() / 255.0).clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)
        both = np.concatenate([gt, img], axis=1)
        Image.fromarray((both * 255).astype(np.uint8)).save(args.out)
        print('saved', args.out, '(left=GT, right=render)')


if __name__ == '__main__':
    main()
