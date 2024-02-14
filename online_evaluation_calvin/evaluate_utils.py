from typing import Dict, Any
import os
from pathlib import Path
import contextlib
from collections import Counter
import glob

import numpy as np
from numpy import pi
import torch
import torch.nn.functional as F
import utils.pytorch3d_transforms as pytorch3d_transforms
import pybullet
import hydra

import calvin_env
from calvin_env.envs.play_table_env import PlayTableSimEnv
from utils.utils_with_calvin import (
    deproject,
    get_gripper_camera_view_matrix,
    convert_rotation
)


############################################################
# Functions to prepare inputs/outputs of 3D diffuser Actor #
############################################################
def prepare_visual_states(obs: Dict[str, Dict[str, Any]],
                          env: PlayTableSimEnv):

    """Prepare point cloud given RGB-D observations.  In-place add point clouds
    to the observation dictionary.

    Args:
        obs: a dictionary of observations
            - rgb_obs: a dictionary of RGB images
            - depth_obs: a dictionary of depth images
            - robot_obs: a dictionary of proprioceptive states
        env: a PlayTableSimEnv instance which contains camera information
    """
    rgb_static = obs["rgb_obs"]["rgb_static"]
    rgb_gripper = obs["rgb_obs"]["rgb_gripper"]
    depth_static = obs["depth_obs"]["depth_static"]
    depth_gripper = obs["depth_obs"]["depth_gripper"]

    static_cam = env.cameras[0]
    gripper_cam = env.cameras[1]
    gripper_cam.viewMatrix = get_gripper_camera_view_matrix(gripper_cam)

    static_pcd = deproject(
        static_cam, depth_static,
        homogeneous=False, sanity_check=False
    ).transpose(1, 0)
    static_pcd = np.reshape(
        static_pcd, (depth_static.shape[0], depth_static.shape[1], 3)
    )
    gripper_pcd = deproject(
        gripper_cam, depth_gripper,
        homogeneous=False, sanity_check=False
    ).transpose(1, 0)
    gripper_pcd = np.reshape(
        gripper_pcd, (depth_gripper.shape[0], depth_gripper.shape[1], 3)
    )

    # map RGB to [0, 1]
    rgb_static = rgb_static / 255.
    rgb_gripper = rgb_gripper / 255.

    h, w = rgb_static.shape[:2]
    rgb_gripper = F.interpolate(
        torch.as_tensor(rgb_gripper).permute(2, 0, 1).unsqueeze(0),
        size=(h, w), mode='bilinear', align_corners=False
    ).squeeze(0).permute(1, 2, 0).numpy()
    gripper_pcd = F.interpolate(
        torch.as_tensor(gripper_pcd).permute(2, 0, 1).unsqueeze(0),
        size=(h, w), mode='nearest'
    ).squeeze(0).permute(1, 2, 0).numpy()

    obs["rgb_obs"]["rgb_static"] = rgb_static
    obs["rgb_obs"]["rgb_gripper"] = rgb_gripper
    obs["pcd_obs"] = {}
    obs["pcd_obs"]["pcd_static"] = static_pcd
    obs["pcd_obs"]["pcd_gripper"] = gripper_pcd

    return obs


def prepare_proprio_states(obs: Dict[str, Dict[str, Any]],
                           env: PlayTableSimEnv):
    """Prepare robot proprioceptive states.  In-place add proprioceptive states
    to the observation dictionary.

    Args:
        obs: a dictionary of observations
            - rgb_obs: a dictionary of RGB images
            - depth_obs: a dictionary of depth images
            - robot_obs: a dictionary of proprioceptive states
        env: a PlayTableSimEnv instance which contains camera information
    """
    # Map gripper openess to [0, 1]
    proprio = np.concatenate([
        obs['robot_obs'][:3],
        convert_rotation(obs['robot_obs'][3:6]),
        (obs['robot_obs'][[-1]] + 1) / 2
    ], axis=-1)

    if 'proprio' not in obs:
        obs['proprio'] = np.stack([proprio] * 3, axis=0)
    else:
        obs['proprio'] = np.concatenate([obs['proprio'][1:], proprio[None]], axis=0)

    return obs


def convert_quaternion_to_euler(quat):
    """Convert Euler angles to Quarternion
    """
    quat = torch.as_tensor(quat)
    mat = pytorch3d_transforms.quaternion_to_matrix(quat)
    rot = pytorch3d_transforms.matrix_to_euler_angles(mat, "XYZ")
    rot = rot.data.cpu().numpy()

    return rot


def convert_action(trajectory):
    """Convert [position, rotation, openess] to the same format as Calvin

    Args:
        trajectory: a torch.Tensor or np.ndarray of shape [bs, traj_len, 8]
            - position: absolute [x, y, z] in the world coordinates
            - rotation: absolute quarternion in the world coordinates
            - openess: [0, 1]

    Returns:
        trajectory: a torch.Tensor or np.ndarray of shape [bs, traj_len, 8]
            - position: absolute [x, y, z] in the world coordinates
            - rotation: absolute 'XYZ' Euler angles in the world coordinates
            - openess: [-1, 1]
    """
    assert trajectory.shape[-1] == 8
    position, rotation, openess = (
        trajectory[..., :3], trajectory[..., 3:7], trajectory[..., -1:]
    )
    position = position.data.cpu().numpy()
    _rot = convert_quaternion_to_euler(rotation)
    # pytorch3d.transforms does not deal with Gumbel lock, the conversion
    # of some rotation matrix results in nan values.  We usepybullet's
    # implementation in this case.
    if (_rot != _rot).any():
        # Pybullet has different convention of Quaternion.
        _rot_shape = list(rotation.shape)[:-1] + [3]
        _rot = rotation.reshape(-1, 4).data.cpu().numpy()
        rotation = np.array([
            pybullet.getEulerFromQuaternion([r[-1], r[0], r[1], r[2]])
            for r in _rot
        ]).reshape(_rot_shape)
    else:
        rotation = _rot
    openess = (2 * (openess >= 0.5).long() - 1).data.cpu().numpy()

    trajectory = np.concatenate([position, rotation, openess], axis=-1)
    return trajectory


######################################################
#     Functions in calvin_agent.evaluation.utils     #
######################################################
def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def get_env(dataset_path, obs_space=None, show_gui=True, **kwargs):
    from pathlib import Path

    from omegaconf import OmegaConf

    render_conf = OmegaConf.load(Path(dataset_path) / ".hydra" / "merged_config.yaml")

    if obs_space is not None:
        exclude_keys = set(render_conf.cameras.keys()) - {
            re.split("_", key)[1] for key in obs_space["rgb_obs"] + obs_space["depth_obs"]
        }
        for k in exclude_keys:
            del render_conf.cameras[k]
    if "scene" in kwargs:
        scene_cfg = OmegaConf.load(Path(calvin_env.__file__).parents[1] / "conf/scene" / f"{kwargs['scene']}.yaml")
        OmegaConf.update(render_conf, "scene", scene_cfg)
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(".")
    env = hydra.utils.instantiate(render_conf.env, show_gui=show_gui, use_vr=False, use_scene_info=True)
    return env


def get_env_state_for_initial_condition(initial_condition):
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ]
    )
    block_rot_z_range = (pi / 2 - pi / 8, pi / 2 + pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    # we want to have a "deterministic" random seed for each initial condition
    import pyhash
    hasher = pyhash.fnv1_32()
    seed = hasher(str(initial_condition.values()))
    with temp_seed(seed):
        np.random.shuffle(block_table)

        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        # red block
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        # blue block
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        # pink block
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)

    return robot_obs, scene_obs


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_log_dir(log_dir):
    if log_dir is not None:
        log_dir = Path(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = Path(__file__).parents[3] / "evaluation"
        if not log_dir.exists():
            log_dir = Path("/tmp/evaluation")
            os.makedirs(log_dir, exist_ok=True)
    print(f"logging to {log_dir}")
    return log_dir


######################################################
#      Functions to cache the evaluation results     #
######################################################
def collect_results(log_dir):
    """Load the number of completed tasks of each instruction chain from a file.
    """
    if os.path.isfile(str(Path(log_dir) / "result.txt")):
        with open(str(Path(log_dir) / "result.txt")) as f:
            lines = f.read().split("\n")[:-1]
    else:
        lines = []

    results, seq_inds = [], []
    for line in lines:
        seq, res= line.split(" ")
        results.append(int(res))
        seq_inds.append(int(seq))

    return results, seq_inds


def write_results(log_dir, seq_ind, result):
    """Write the number of completed tasks of each instruction chain to a file.
    """
    with open(log_dir / f"result.txt", "a") as write_file:
        write_file.write(f"{seq_ind} {result}\n")