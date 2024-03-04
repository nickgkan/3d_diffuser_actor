"""Main script for keypose optimization."""

import os
from pathlib import Path
import random
from typing import Tuple, Optional

import numpy as np
import tap
import torch
import torch.distributed as dist
from torch.nn import functional as F

from datasets.dataset_engine import RLBenchDataset
from engine import BaseTrainTester
from diffuser_actor import Act3D
from utils.common_utils import (
    load_instructions, count_parameters, get_gripper_loc_bounds
)


class Arguments(tap.Tap):
    cameras: Tuple[str, ...] = ("wrist", "left_shoulder", "right_shoulder")
    image_size: str = "256,256"
    max_episodes_per_task: int = 100
    instructions: Optional[Path] = "instructions.pkl"
    seed: int = 0
    tasks: Tuple[str, ...]
    variations: Tuple[int, ...] = (0,)
    checkpoint: Optional[Path] = None
    accumulate_grad_batches: int = 1
    val_freq: int = 500
    gripper_loc_bounds: Optional[str] = None
    gripper_loc_bounds_buffer: float = 0.04
    eval_only: int = 0

    # Training and validation datasets
    dataset: Path
    valset: Path

    # Logging to base_log_dir/exp_log_dir/run_log_dir
    base_log_dir: Path = Path(__file__).parent / "train_logs"
    exp_log_dir: str = "exp"
    run_log_dir: str = "run"

    # Main training parameters
    num_workers: int = 1
    batch_size: int = 16
    batch_size_val: int = 4
    cache_size: int = 100
    cache_size_val: int = 100
    lr: float = 1e-4
    train_iters: int = 200_000
    max_episode_length: int = 5  # -1 for no limit

    # Data augmentations
    image_rescale: str = "0.75,1.25"  # (min, max), "1.0,1.0" for no rescaling

    # Loss
    position_loss: str = "ce"  # one of "ce" (our model), "mse" (HiveFormer)
    ground_truth_gaussian_spread: float = 0.01
    compute_loss_at_all_layers: int = 0
    position_loss_coeff: float = 1.0
    position_offset_loss_coeff: float = 10000.0
    rotation_loss_coeff: float = 10.0
    symmetric_rotation_loss: int = 0
    gripper_loss_coeff: float = 1.0
    label_smoothing: float = 0.0
    regress_position_offset: int = 0

    # Ghost points
    num_sampling_level: int = 3
    fine_sampling_ball_diameter: float = 0.16
    weight_tying: int = 1
    gp_emb_tying: int = 1
    num_ghost_points: int = 1000
    num_ghost_points_val: int = 10000
    use_ground_truth_position_for_sampling_train: int = 1  # considerably speeds up training

    # Model
    action_dim: int = 8
    backbone: str = "clip"  # one of "resnet", "clip"
    embedding_dim: int = 120
    num_ghost_point_cross_attn_layers: int = 2
    num_query_cross_attn_layers: int = 2
    num_vis_ins_attn_layers: int = 2
    rotation_parametrization: str = "quat_from_query"
    use_instruction: int = 0


class TrainTester(BaseTrainTester):
    """Train/test a keypose optimization algorithm."""

    def __init__(self, args):
        """Initialize."""
        super().__init__(args)

    def get_datasets(self):
        """Initialize datasets."""
        # Load instruction, based on which we load tasks/variations
        instruction = load_instructions(
            self.args.instructions,
            tasks=self.args.tasks,
            variations=self.args.variations
        )
        if instruction is None:
            raise NotImplementedError()
        else:
            taskvar = [
                (task, var)
                for task, var_instr in instruction.items()
                for var in var_instr.keys()
            ]

        # Initialize datasets with arguments
        train_dataset = RLBenchDataset(
            root=self.args.dataset,
            instructions=instruction,
            taskvar=taskvar,
            max_episode_length=self.args.max_episode_length,
            cache_size=self.args.cache_size,
            max_episodes_per_task=self.args.max_episodes_per_task,
            num_iters=self.args.train_iters,
            cameras=self.args.cameras,
            training=True,
            image_rescale=tuple(
                float(x) for x in self.args.image_rescale.split(",")
            ),
            return_low_lvl_trajectory=False,
            dense_interpolation=False,
            interpolation_length=0
        )
        test_dataset = RLBenchDataset(
            root=self.args.valset,
            instructions=instruction,
            taskvar=taskvar,
            max_episode_length=self.args.max_episode_length,
            cache_size=self.args.cache_size_val,
            max_episodes_per_task=self.args.max_episodes_per_task,
            cameras=self.args.cameras,
            training=False,
            image_rescale=tuple(
                float(x) for x in self.args.image_rescale.split(",")
            ),
            return_low_lvl_trajectory=False,
            dense_interpolation=False,
            interpolation_length=0
        )
        return train_dataset, test_dataset

    def get_model(self):
        """Initialize the model."""
        # Initialize model with arguments
        args = self.args
        _model = Act3D(
            backbone=args.backbone,
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=args.embedding_dim,
            num_ghost_point_cross_attn_layers=args.num_ghost_point_cross_attn_layers,
            num_query_cross_attn_layers=args.num_query_cross_attn_layers,
            num_vis_ins_attn_layers=args.num_vis_ins_attn_layers,
            rotation_parametrization=args.rotation_parametrization,
            gripper_loc_bounds=self.args.gripper_loc_bounds,
            num_ghost_points=args.num_ghost_points,
            num_ghost_points_val=args.num_ghost_points_val,
            weight_tying=bool(args.weight_tying),
            gp_emb_tying=bool(args.gp_emb_tying),
            num_sampling_level=args.num_sampling_level,
            fine_sampling_ball_diameter=args.fine_sampling_ball_diameter,
            regress_position_offset=bool(args.regress_position_offset),
            use_instruction=bool(args.use_instruction)
        )
        print("Model parameters:", count_parameters(_model))

        return _model

    def get_criterion(self):
        args = self.args
        return LossAndMetrics(
            rotation_parametrization=args.rotation_parametrization,
            position_loss=args.position_loss,
            compute_loss_at_all_layers=bool(args.compute_loss_at_all_layers),
            ground_truth_gaussian_spread=args.ground_truth_gaussian_spread,
            label_smoothing=args.label_smoothing,
            position_loss_coeff=args.position_loss_coeff,
            position_offset_loss_coeff=args.position_offset_loss_coeff,
            rotation_loss_coeff=args.rotation_loss_coeff,
            gripper_loss_coeff=args.gripper_loss_coeff,
            symmetric_rotation_loss=bool(args.symmetric_rotation_loss)
        )

    def train_one_step(self, model, criterion, optimizer, step_id, sample):
        """Run a single training step."""
        if step_id % self.args.accumulate_grad_batches == 0:
            optimizer.zero_grad()

        # Forward pass
        out = model(
            sample["rgbs"],
            sample["pcds"],
            sample["instr"],
            sample["curr_gripper"],
            # Provide ground-truth action to bias ghost point sampling at training time
            gt_action=sample["action"] if self.args.use_ground_truth_position_for_sampling_train else None
        )

        # Backward pass
        loss = criterion.compute_loss(out, sample)
        loss = sum(list(loss.values()))
        loss.backward()

        # Update
        if step_id % self.args.accumulate_grad_batches == self.args.accumulate_grad_batches - 1:
            optimizer.step()

        # Log
        if dist.get_rank() == 0 and (step_id + 1) % self.args.val_freq == 0:
            self.writer.add_scalar("lr", self.args.lr, step_id)
            self.writer.add_scalar("train-loss/noise_mse", loss, step_id)

    @torch.no_grad()
    def evaluate_nsteps(self, model, criterion, loader, step_id, val_iters,
                        split='val'):
        """Run a given number of evaluation steps."""
        values = {}
        device = next(model.parameters()).device
        model.eval()

        for i, sample in enumerate(loader):
            if i == val_iters:
                break

            action = model(
                sample["rgbs"],
                sample["pcds"],
                sample["instr"],
                sample["curr_gripper"],
                # DO NOT provide ground-truth action to sample ghost points at validation time
                gt_action=None
            )
            losses = criterion.compute_metrics(
                action,
                sample
            )

            # Gather global statistics
            for n, l in losses.items():
                key = f"{split}-losses/{n}"
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

        # Log all statistics
        values = {
            k: torch.as_tensor(v).mean().item() for k, v in values.items()
        }
        if dist.get_rank() == 0:
            for key, val in values.items():
                self.writer.add_scalar(key, val, step_id)

            # Also log to terminal
            print(f"Step {step_id}:")
            for key, value in values.items():
                print(f"{key}: {value:.03f}")

        return values.get('val-losses/action_mse', None)


def keypose_collate_fn(batch):
    # Unfold multi-step demos to form a longer batch
    keys = ["rgbs", "pcds", "curr_gripper", "action", "instr"]
    ret_dict = {key: torch.cat([item[key] for item in batch]) for key in keys}

    ret_dict["task"] = []
    for item in batch:
        ret_dict["task"] += item['task']
    return ret_dict


class LossAndMetrics:
    """
    Each method expects two dictionaries:
     - pred: {
        'position': (B, 3) gripper position,
        'rotation': (B, 4) gripper rotation,
        'gripper': (B, 1) whether gripper should open/close (0/1),
        'position_pyramid': list of 3 elements, (B, 1, 3) interm gripper pos,
        'visible_rgb_mask_pyramid': not used in loss,
        'ghost_pcd_masks_pyramid',
        'ghost_pcd_pyramid',
        'fine_ghost_pcd_offsets',
        'task'
     }
     - sample: {
        'frame_id',
        'task_id',
        'task',
        'variation',
        'rgbs',
        'pcds',
        'action': (B, 1, 8),
        'padding_mask': (B, 1),
        'instr',
        'gripper'
     }
    """
    def __init__(
        self,
        position_loss,
        rotation_parametrization,
        ground_truth_gaussian_spread,
        compute_loss_at_all_layers=False,
        label_smoothing=0.0,
        position_loss_coeff=1.0,
        position_offset_loss_coeff=10000.0,
        rotation_loss_coeff=10.0,
        gripper_loss_coeff=1.0,
        symmetric_rotation_loss=False,
    ):
        assert position_loss in ["mse", "ce", "ce+mse"]
        assert rotation_parametrization in [
            "quat_from_top_ghost", "quat_from_query",
            "6D_from_top_ghost", "6D_from_query"
        ]
        self.position_loss = position_loss
        self.rotation_parametrization = rotation_parametrization
        self.compute_loss_at_all_layers = compute_loss_at_all_layers
        self.ground_truth_gaussian_spread = ground_truth_gaussian_spread
        self.label_smoothing = label_smoothing
        self.position_loss_coeff = position_loss_coeff
        self.position_offset_loss_coeff = position_offset_loss_coeff
        self.rotation_loss_coeff = rotation_loss_coeff
        self.gripper_loss_coeff = gripper_loss_coeff
        self.symmetric_rotation_loss = symmetric_rotation_loss

    def compute_loss(self, pred, sample):
        device = pred["position"].device
        # padding_mask = sample["padding_mask"].to(device)
        gt_action = sample["action"].to(device)  # [padding_mask]

        losses = {}

        self._compute_position_loss(pred, gt_action[:, :3], losses)

        self._compute_rotation_loss(pred, gt_action[:, 3:7], losses)

        losses["gripper"] = F.binary_cross_entropy(pred["gripper"], gt_action[:, 7:8])
        losses["gripper"] *= self.gripper_loss_coeff

        return losses

    def _compute_rotation_loss(self, pred, gt_quat, losses):
        if "quat" in self.rotation_parametrization:
            if self.symmetric_rotation_loss:
                gt_quat_ = -gt_quat.clone()
                quat_loss = F.mse_loss(pred["rotation"], gt_quat, reduction='none').mean(1)
                quat_loss_ = F.mse_loss(pred["rotation"], gt_quat_, reduction='none').mean(1)
                select_mask = (quat_loss < quat_loss_).float()
                losses['rotation'] = (select_mask * quat_loss + (1 - select_mask) * quat_loss_).mean()
            else:
                losses["rotation"] = F.mse_loss(pred["rotation"], gt_quat)

        losses["rotation"] *= self.rotation_loss_coeff

    def _compute_position_loss(self, pred, gt_position, losses):
        if self.position_loss == "mse":
            # Only used for original HiveFormer
            losses["position_mse"] = F.mse_loss(pred["position"], gt_position) * self.position_loss_coeff

        elif self.position_loss in ["ce", "ce+mse"]:
            # Select a normalized Gaussian ball around the ground-truth
            # as a proxy label for a soft cross-entropy loss
            l2_pyramid = []
            label_pyramid = []
            for ghost_pcd_i in pred['ghost_pcd_pyramid']:
                l2_i = ((ghost_pcd_i - gt_position.unsqueeze(-1)) ** 2).sum(1).sqrt()
                label_i = torch.softmax(-l2_i / self.ground_truth_gaussian_spread, dim=-1).detach()
                l2_pyramid.append(l2_i)
                label_pyramid.append(label_i)

            loss_layers = range(len(pred['ghost_pcd_masks_pyramid'][0])) if self.compute_loss_at_all_layers else [-1]

            for j in loss_layers:
                for i, ghost_pcd_masks_i in enumerate(pred["ghost_pcd_masks_pyramid"]):
                    losses[f"position_ce_level{i}"] = F.cross_entropy(
                        ghost_pcd_masks_i[j], label_pyramid[i],
                        label_smoothing=self.label_smoothing
                    ).mean() * self.position_loss_coeff / len(pred["ghost_pcd_masks_pyramid"])

            # Supervise offset from the ghost point's position to the predicted position
            num_sampling_level = len(pred['ghost_pcd_masks_pyramid'])
            if pred.get("fine_ghost_pcd_offsets") is not None:
                if pred["ghost_pcd_pyramid"][-1].shape[-1] != pred["ghost_pcd_pyramid"][0].shape[-1]:
                    npts = pred["ghost_pcd_pyramid"][-1].shape[-1] // num_sampling_level
                    pred_with_offset = (pred["ghost_pcd_pyramid"][-1] + pred["fine_ghost_pcd_offsets"])[:, :, -npts:]
                else:
                    pred_with_offset = (pred["ghost_pcd_pyramid"][-1] + pred["fine_ghost_pcd_offsets"])
                losses["position_offset"] = F.mse_loss(
                    pred_with_offset,
                    gt_position.unsqueeze(-1).repeat(1, 1, pred_with_offset.shape[-1])
                )
                losses["position_offset"] *= (self.position_offset_loss_coeff * self.position_loss_coeff)

            if self.position_loss == "ce":
                # Clear gradient on pred["position"] to avoid a memory leak since we don't
                # use it in the loss
                pred["position"] = pred["position"].detach()
            else:
                losses["position_mse"] = (
                    F.mse_loss(pred["position"], gt_position)
                    * self.position_loss_coeff
                )

    def compute_metrics(self, pred, sample):
        device = pred["position"].device
        dtype = pred["position"].dtype
        # padding_mask = sample["padding_mask"].to(device)
        outputs = sample["action"].to(device)  # [padding_mask]

        metrics = {}

        tasks = np.array(sample["task"])

        final_pos_l2 = ((pred["position"] - outputs[:, :3]) ** 2).sum(1).sqrt()
        metrics["mean/pos_l2_final"] = final_pos_l2.to(dtype).mean()
        metrics["mean/pos_l2_final<0.01"] = (final_pos_l2 < 0.01).to(dtype).mean()

        for i in range(len(pred["position_pyramid"])):
            pos_l2_i = ((pred["position_pyramid"][i].squeeze(1) - outputs[:, :3]) ** 2).sum(1).sqrt()
            metrics[f"mean/pos_l2_level{i}"] = pos_l2_i.to(dtype).mean()

        for task in np.unique(tasks):
            task_l2 = final_pos_l2[tasks == task]
            metrics[f"{task}/pos_l2_final"] = task_l2.to(dtype).mean()
            metrics[f"{task}/pos_l2_final<0.01"] = (task_l2 < 0.01).to(dtype).mean()

        # Gripper accuracy
        pred_gripper = (pred["gripper"] > 0.5).squeeze(-1)
        true_gripper = outputs[:, 7].bool()
        acc = pred_gripper == true_gripper
        metrics["gripper"] = acc.to(dtype).mean()

        # Rotation accuracy
        gt_quat = outputs[:, 3:7]
        if "quat" in self.rotation_parametrization:
            if self.symmetric_rotation_loss:
                gt_quat_ = -gt_quat.clone()
                l1 = (pred["rotation"] - gt_quat).abs().sum(1)
                l1_ = (pred["rotation"] - gt_quat_).abs().sum(1)
                select_mask = (l1 < l1_).float()
                l1 = (select_mask * l1 + (1 - select_mask) * l1_)
            else:
                l1 = ((pred["rotation"] - gt_quat).abs().sum(1))

        metrics["mean/rot_l1"] = l1.to(dtype).mean()
        metrics["mean/rot_l1<0.05"] = (l1 < 0.05).to(dtype).mean()
        metrics["mean/rot_l1<0.025"] = (l1 < 0.025).to(dtype).mean()

        for task in np.unique(tasks):
            task_l1 = l1[tasks == task]
            metrics[f"{task}/rot_l1"] = task_l1.to(dtype).mean()
            metrics[f"{task}/rot_l1<0.05"] = (task_l1 < 0.05).to(dtype).mean()
            metrics[f"{task}/rot_l1<0.025"] = (task_l1 < 0.025).to(dtype).mean()

        return metrics


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Arguments
    args = Arguments().parse_args()
    print("Arguments:")
    print(args)
    print("-" * 100)
    if args.gripper_loc_bounds is None:
        args.gripper_loc_bounds = np.array([[-2, -2, -2], [2, 2, 2]]) * 1.0
    else:
        args.gripper_loc_bounds = get_gripper_loc_bounds(
            args.gripper_loc_bounds,
            task=args.tasks[0] if len(args.tasks) == 1 else None,
            buffer=args.gripper_loc_bounds_buffer
        )
    log_dir = args.base_log_dir / args.exp_log_dir / args.run_log_dir
    args.log_dir = log_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    print("Logging:", log_dir)
    print(
        "Available devices (CUDA_VISIBLE_DEVICES):",
        os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    print("Device count", torch.cuda.device_count())
    args.local_rank = int(os.environ["LOCAL_RANK"])

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # DDP initialization
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Run
    train_tester = TrainTester(args)
    train_tester.main(collate_fn=keypose_collate_fn)
