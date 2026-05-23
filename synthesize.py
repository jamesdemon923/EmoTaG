import copy
import os
import traceback
from argparse import ArgumentParser
from os import makedirs

import imageio
import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render_motion
from scene import GaussianModel, MotionNetwork, Scene
from scene.flame_wrapper import SimpleFlameWrapper
from utils.general_utils import safe_state


def render_set(model_path, name, views, gaussians, motion_net, pipeline, background):
    """Render a camera split and save rendered/ground-truth videos."""
    output_dir = os.path.join(model_path, name)
    makedirs(output_dir, exist_ok=True)

    rendered_frames = []
    gt_frames = []
    print(f"Rendering {len(views)} frames...")

    for view in tqdm(views, desc="Rendering", ascii=True):
        with torch.no_grad():
            render_pkg = render_motion(view, gaussians, motion_net, pipeline, background)
            rendered_image = render_pkg["render"]
            rendered_frame = (rendered_image.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            gt_frame = (view.original_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            rendered_frames.append(rendered_frame)
            gt_frames.append(gt_frame)

    rendered_video_path = os.path.join(output_dir, "rendered_video.mp4")
    gt_video_path = os.path.join(output_dir, "gt_video.mp4")
    imageio.mimwrite(rendered_video_path, rendered_frames, fps=25, quality=8, macro_block_size=1)
    imageio.mimwrite(gt_video_path, gt_frames, fps=25, quality=8, macro_block_size=1)

    print(f"Rendered video saved to: {rendered_video_path}")
    print(f"Ground-truth video saved to: {gt_video_path}")
    return rendered_video_path, gt_video_path


def render_sets(dataset: ModelParams, pipeline: PipelineParams, use_train: bool):
    """Load a trained EmoTaG scene and render train or validation cameras."""
    with torch.no_grad():
        dataset.type = "face"
        gaussians = GaussianModel(copy.deepcopy(dataset))

        flame_params_file = os.path.join(dataset.source_path, "flame_params.npz")
        if not os.path.exists(flame_params_file):
            raise FileNotFoundError(f"Integrated FLAME parameters file not found: {flame_params_file}")

        model_center_path = os.path.join(dataset.source_path, "model_center.npy")
        if not os.path.exists(model_center_path):
            raise FileNotFoundError(f"Model center file not found: {model_center_path}")

        dataset.model_center = np.load(model_center_path)
        gaussians.flame_wrapper = SimpleFlameWrapper(flame_params_file=flame_params_file).cuda()
        scene = Scene(dataset, gaussians, shuffle=False)
        motion_net = MotionNetwork(args=dataset).cuda()

        checkpoint_path = os.path.join(dataset.model_path, "chkpnt_face_latest.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint from: {checkpoint_path}")
        model_params, motion_params, _, _ = torch.load(checkpoint_path)
        motion_net.load_state_dict(motion_params, strict=False)
        gaussians.restore(model_params, None)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        dataset_name = "train" if use_train else "test"
        views = scene.getTrainCameras() if use_train else scene.getTestCameras()
        return render_set(dataset.model_path, dataset_name, views, gaussians, motion_net, pipeline, background)


if __name__ == "__main__":
    parser = ArgumentParser(description="Video synthesis script for EmoTaG")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--use_train", action="store_true", help="Use training set instead of test set")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = get_combined_args(parser)

    print("=" * 70)
    print("EmoTaG Video Synthesis")
    print("=" * 70)
    print(f"Model path: {args.model_path}")
    print(f"Dataset split: {'train' if args.use_train else 'test'}")
    print("=" * 70)

    safe_state(args.quiet)
    try:
        rendered_video_path, gt_video_path = render_sets(model.extract(args), pipeline.extract(args), args.use_train)
        output_video_dir = os.path.dirname(rendered_video_path)
        print("\nSynthesis completed successfully.")
        print(f"Rendered video: {rendered_video_path}")
        print(f"Ground-truth video: {gt_video_path}")
        print("Next step:")
        print(f"  python evaluate_metrics.py video {rendered_video_path} {gt_video_path} {output_video_dir}")
    except Exception as exc:
        print(f"Error during synthesis: {exc}")
        traceback.print_exc()
        raise SystemExit(1)
