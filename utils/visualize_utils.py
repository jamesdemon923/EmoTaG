"""Lightweight training logging helpers for EmoTaG."""

import json
import os

import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image


class LossTracker:
    """Track per-scene training losses and write simple text logs."""

    def __init__(self, output_dir, data_list, warm_up_iter, scene_paths):
        self.output_dir = output_dir
        self.data_list = data_list
        self.warm_up_iter = warm_up_iter
        self.scene_paths = scene_paths
        self.scene_losses = {scene_name: [] for scene_name in data_list}
        self.detailed_loss_path = os.path.join(output_dir, "training_loss.txt")
        self.scene_loss_files = {}
        os.makedirs(output_dir, exist_ok=True)
        self._initialize_detailed_loss_file()
        for scene_name in data_list:
            scene_path = scene_paths[scene_name]
            os.makedirs(scene_path, exist_ok=True)
            scene_loss_file = os.path.join(scene_path, "training_loss.txt")
            self.scene_loss_files[scene_name] = scene_loss_file
            self._initialize_scene_loss_file(scene_loss_file)

    def _initialize_detailed_loss_file(self):
        with open(self.detailed_loss_path, "w", encoding="utf-8") as file:
            file.write("Scene | Iteration | Frame | Loss | Phase\n")

    def _initialize_scene_loss_file(self, scene_loss_file):
        with open(scene_loss_file, "w", encoding="utf-8") as file:
            file.write("Iteration | Frame | Loss | Phase\n")

    def record_loss(self, iteration, scene_name, frame_id, loss_value):
        is_warmup = iteration <= self.warm_up_iter
        phase = "warmup" if is_warmup else "dynamic"
        value = float(loss_value)
        self.scene_losses.setdefault(scene_name, []).append((iteration, value, is_warmup, frame_id))
        with open(self.detailed_loss_path, "a", encoding="utf-8") as file:
            file.write(f"{scene_name} | {iteration} | {frame_id} | {value:.6f} | {phase}\n")
        with open(self.scene_loss_files[scene_name], "a", encoding="utf-8") as file:
            file.write(f"{iteration} | {frame_id} | {value:.6f} | {phase}\n")

    def get_scene_loss_history(self, scene_name):
        return [loss for _, loss, _, _ in self.scene_losses.get(scene_name, [])]

    def get_all_scene_loss_histories(self):
        return {scene_name: self.get_scene_loss_history(scene_name) for scene_name in self.data_list}


class ImageDebugger:
    """Save optional visual snapshots and loss curves during training."""

    def __init__(self, output_dir, data_list, warm_up_iter, scene_paths):
        self.output_dir = output_dir
        self.data_list = data_list
        self.warm_up_iter = warm_up_iter
        self.scene_paths = scene_paths
        self.last_renders = {}
        self.loss_tracker = LossTracker(output_dir, data_list, warm_up_iter, scene_paths)

    def save_images_and_update_loss(self, iteration, current_scene_name, rendered_image, gt_image, loss_value, scene_model_path, frame_id, save_images=False):
        image_t = rendered_image.detach().clone()
        gt_image_t = gt_image.detach().clone()
        if save_images:
            image_save_path = os.path.join(scene_model_path, "images")
            os.makedirs(image_save_path, exist_ok=True)
            save_image(image_t, os.path.join(image_save_path, f"render_{iteration}_{frame_id}.png"))
            save_image(gt_image_t, os.path.join(image_save_path, f"gt_{iteration}_{frame_id}.png"))
        self.loss_tracker.record_loss(iteration, current_scene_name, frame_id, loss_value)
        self.last_renders[current_scene_name] = {"render": image_t, "gt": gt_image_t, "scene_model_path": scene_model_path}
        self._update_loss_plot()

    def _update_loss_plot(self):
        histories = self.loss_tracker.get_all_scene_loss_histories()
        for scene_name, losses in histories.items():
            if not losses:
                continue
            plt.figure(figsize=(8, 5))
            plt.plot(losses, label="training loss", linewidth=2)
            plt.xlabel("Scene updates")
            plt.ylabel("Loss")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            scene_path = self.scene_paths[scene_name]
            plt.savefig(os.path.join(scene_path, "loss_curve.png"), dpi=120, bbox_inches="tight")
            plt.close()

    def save_final_images(self):
        for scene_name, data in self.last_renders.items():
            image_save_path = os.path.join(data["scene_model_path"], "images")
            os.makedirs(image_save_path, exist_ok=True)
            save_image(data["render"], os.path.join(image_save_path, "render_final.png"))
            save_image(data["gt"], os.path.join(image_save_path, "gt_final.png"))
            print(f"Saved final images for scene: {scene_name}")

    def save_debug_comparison(self, gt_image_original, head_mask, gt_image, background):
        mask_visual = head_mask.repeat(1, 3, 1, 1).float()
        if gt_image_original.dim() == 3:
            gt_image_original = gt_image_original.unsqueeze(0)
        if gt_image.dim() == 3:
            gt_image = gt_image.unsqueeze(0)
        comparison_grid = torch.cat([gt_image_original, mask_visual, gt_image], dim=0)
        save_image(comparison_grid, os.path.join(self.output_dir, "pretrain_debug_comparison.png"), nrow=3)

    def save_specific_iteration_images(self, iteration, scene_name, frame_id, rendered_image, gt_image):
        iteration_dir = os.path.join(self.output_dir, "specific_iterations", f"iter_{iteration:06d}")
        os.makedirs(iteration_dir, exist_ok=True)
        save_image(rendered_image.clone(), os.path.join(iteration_dir, f"{scene_name}_{frame_id}_render.png"))
        save_image(gt_image.clone(), os.path.join(iteration_dir, f"{scene_name}_{frame_id}_gt.png"))


class TrainingStatsTracker:
    """Record how often each frame is sampled during training."""

    def __init__(self, data_list, output_dir):
        self.data_list = data_list
        self.output_dir = output_dir
        self.training_stats = {scene_name: {} for scene_name in data_list}

    def initialize_from_scenes(self, scene_list):
        for scene_idx, scene_name in enumerate(self.data_list):
            self.training_stats[scene_name] = {}
            for cam in scene_list[scene_idx].getTrainCameras():
                self.training_stats[scene_name][cam.image_name] = 0

    def record_training(self, scene_name, frame_id):
        self.training_stats.setdefault(scene_name, {})
        self.training_stats[scene_name][frame_id] = self.training_stats[scene_name].get(frame_id, 0) + 1

    def save_intermediate_report(self, iteration):
        report_dir = os.path.join(self.output_dir, "stats_report")
        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, f"stats_{iteration:06d}.json"), "w", encoding="utf-8") as file:
            json.dump(self.training_stats, file, indent=2)

    def save_final_report(self, final_iteration):
        report_path = os.path.join(self.output_dir, "training_statistics.json")
        with open(report_path, "w", encoding="utf-8") as file:
            json.dump({"final_iteration": final_iteration, "stats": self.training_stats}, file, indent=2)
        print(f"Training statistics saved to: {report_path}")


class FlameDebugger:
    """Write compact FLAME prediction diagnostics."""

    def __init__(self, output_dir, debug_save_interval=50):
        self.output_dir = output_dir
        self.debug_save_interval = debug_save_interval
        self.flame_report_dir = os.path.join(output_dir, "flame_report")
        os.makedirs(self.flame_report_dir, exist_ok=True)
        self.exp_debug_path = os.path.join(self.flame_report_dir, "flame_expression_debug.txt")
        self.jaw_debug_path = os.path.join(self.flame_report_dir, "flame_jaw_debug.txt")
        self._initialize_debug_files()

    def _initialize_debug_files(self):
        with open(self.exp_debug_path, "w", encoding="utf-8") as file:
            file.write("iteration | scene | frame | pred_mean | gt_mean | l2 | render_loss | flame_loss\n")
        with open(self.jaw_debug_path, "w", encoding="utf-8") as file:
            file.write("iteration | scene | frame | pred_jaw | gt_jaw | l2 | render_loss | flame_loss\n")

    def record_flame_params(self, iteration, scene_name, frame_id, pred_exp, pred_jaw, gt_exp, gt_jaw, rendering_loss, flame_reg_loss):
        if iteration % self.debug_save_interval != 0 and iteration > 100:
            return
        pred_exp_flat = pred_exp.detach().flatten()[:50]
        gt_exp_flat = gt_exp.detach().flatten()[:50]
        exp_l2 = torch.mean((pred_exp_flat - gt_exp_flat) ** 2).item()
        pred_jaw_flat = pred_jaw.detach().flatten()[:3]
        gt_jaw_flat = gt_jaw.detach().flatten()[:3]
        jaw_l2 = torch.mean((pred_jaw_flat - gt_jaw_flat) ** 2).item()
        with open(self.exp_debug_path, "a", encoding="utf-8") as file:
            file.write(
                f"{iteration} | {scene_name} | {frame_id} | {pred_exp_flat.mean().item():.6f} | "
                f"{gt_exp_flat.mean().item():.6f} | {exp_l2:.6f} | {float(rendering_loss):.6f} | {float(flame_reg_loss):.6f}\n"
            )
        with open(self.jaw_debug_path, "a", encoding="utf-8") as file:
            file.write(
                f"{iteration} | {scene_name} | {frame_id} | {pred_jaw_flat.cpu().tolist()} | "
                f"{gt_jaw_flat.cpu().tolist()} | {jaw_l2:.6f} | {float(rendering_loss):.6f} | {float(flame_reg_loss):.6f}\n"
            )

    def print_summary(self, iteration):
        print(f"FLAME diagnostics are being written every {self.debug_save_interval} iterations; current iteration: {iteration}")

    def generate_final_report(self):
        print(f"FLAME diagnostic logs saved to: {self.flame_report_dir}")
