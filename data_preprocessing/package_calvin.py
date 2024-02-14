from typing import List, Optional
from pathlib import Path
import os
import pickle

import tap
import cv2
import numpy as np
import torch
import blosc
from PIL import Image

from calvin_env.envs.play_table_env import get_env
from utils.utils_with_calvin import (
    keypoint_discovery,
    deproject,
    get_gripper_camera_view_matrix,
    convert_rotation
)


class Arguments(tap.Tap):
    traj_len: int = 16
    execute_every: int = 4
    save_path: str = './data/calvin/packaged_ABC_D'
    root_dir: str = './calvin/dataset/task_ABC_D'
    mode: str = 'keypose'  # [keypose, close_loop]
    tasks: Optional[List[str]] = None
    split: str = 'training'  # [training, validation]


def make_env(dataset_path, split):
    val_folder = Path(dataset_path) / f"{split}"
    env = get_env(val_folder, show_gui=False)

    return env


def process_datas(datas, mode, traj_len, execute_every, keyframe_inds):
    """Fetch and drop datas to make a trajectory

    Args:
        datas: a dict of the datas to be saved/loaded
            - static_pcd: a list of nd.arrays with shape (height, width, 3)
            - static_rgb: a list of nd.arrays with shape (height, width, 3)
            - gripper_pcd: a list of nd.arrays with shape (height, width, 3)
            - gripper_rgb: a list of nd.arrays with shape (height, width, 3)
            - proprios: a list of nd.arrays with shape (7,)
        mode: a string of [keypose, close_loop]
        traj_len: an int of the length of the trajectory
        execute_every: an int of execution frequency
        keyframe_inds: an Integer array with shape (num_keyframes,)

    Returns:
        the episode item: [
            [frame_ids],
            [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256)
                obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
            [action_tensors],  # wrt frame_ids, (1, 8)
            [camera_dicts],
            [gripper_tensors],  # wrt frame_ids, (1, 8)
            [trajectories]  # wrt frame_ids, (N_i, 8)
            [annotation_ind] # wrt frame_ids, (1,)
        ]
    """
    # upscale gripper camera
    h, w = datas['static_rgb'][0].shape[:2]
    datas['gripper_rgb'] = [
        cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        for m in datas['gripper_rgb']
    ]
    datas['gripper_pcd'] = [
        cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        for m in datas['gripper_pcd']
    ]
    static_rgb = np.stack(datas['static_rgb'], axis=0) # (traj_len, H, W, 3)
    static_pcd = np.stack(datas['static_pcd'], axis=0) # (traj_len, H, W, 3)
    gripper_rgb = np.stack(datas['gripper_rgb'], axis=0) # (traj_len, H, W, 3)
    gripper_pcd = np.stack(datas['gripper_pcd'], axis=0) # (traj_len, H, W, 3)
    rgb = np.stack([static_rgb, gripper_rgb], axis=1) # (traj_len, ncam, H, W, 3)
    pcd = np.stack([static_pcd, gripper_pcd], axis=1) # (traj_len, ncam, H, W, 3)
    rgb_pcd = np.stack([rgb, pcd], axis=2) # (traj_len, ncam, 2, H, W, 3)])
    rgb_pcd = rgb_pcd.transpose(0, 1, 2, 5, 3, 4) # (traj_len, ncam, 2, 3, H, W)
    rgb_pcd = torch.as_tensor(rgb_pcd, dtype=torch.float32) # (traj_len, ncam, 2, 3, H, W)

    # prepare keypose actions
    keyframe_indices = torch.as_tensor(keyframe_inds)[None, :]
    gripper_indices = torch.arange(len(datas['proprios'])).view(-1, 1)
    action_indices = torch.argmax(
        (gripper_indices < keyframe_indices).float(), dim=1
    ).tolist()
    action_indices[-1] = len(keyframe_inds) - 1
    actions = [datas['proprios'][keyframe_inds[i]] for i in action_indices]
    action_tensors = [
        torch.as_tensor(a, dtype=torch.float32).view(1, -1) for a in actions
    ]

    # prepare camera_dicts
    camera_dicts = [{'front': (0, 0), 'wrist': (0, 0)}]

    # prepare gripper tensors
    gripper_tensors = [
        torch.as_tensor(a, dtype=torch.float32).view(1, -1)
        for a in datas['proprios']
    ]

    # prepare trajectories
    if mode == 'keypose':
        trajectories = []
        for i in range(len(action_indices)):
            target_frame = keyframe_inds[action_indices[i]]
            current_frame = i
            trajectories.append(
                torch.cat(
                    [
                        torch.as_tensor(a, dtype=torch.float32).view(1, -1)
                        for a in datas['proprios'][current_frame:target_frame+1]
                    ],
                    dim=0
                )
            )
    else:
        trajectories = []
        for i in range(len(gripper_tensors)):
            traj = datas['proprios'][i:i+traj_len]
            if len(traj) < traj_len:
                traj += [traj[-1]] * (traj_len - len(traj))
            traj = [
                torch.as_tensor(a, dtype=torch.float32).view(1, -1)
                for a in traj
            ]
            traj = torch.cat(traj, dim=0)
            trajectories.append(traj)

    # Filter out datas
    if mode == 'keypose':
        keyframe_inds = [0] + keyframe_inds[:-1].tolist()
        keyframe_indices = torch.as_tensor(keyframe_inds)
        rgb_pcd = torch.index_select(rgb_pcd, 0, keyframe_indices)
        action_tensors = [action_tensors[i] for i in keyframe_inds]
        gripper_tensors = [gripper_tensors[i] for i in keyframe_inds]
        trajectories = [trajectories[i] for i in keyframe_inds]
    else:
        rgb_pcd = rgb_pcd[:-1]
        action_tensors = action_tensors[:-1]
        gripper_tensors = gripper_tensors[:-1]
        trajectories = trajectories[:-1]

        rgb_pcd = rgb_pcd[::execute_every]
        action_tensors = action_tensors[::execute_every]
        gripper_tensors = gripper_tensors[::execute_every]
        trajectories = trajectories[::execute_every]

    # prepare frame_ids
    frame_ids = [i for i in range(len(rgb_pcd))]

    # Save everything to disk
    state_dict = [
        frame_ids,
        rgb_pcd,
        action_tensors,
        camera_dicts,
        gripper_tensors,
        trajectories,
        datas['annotation_id']
    ]

    return state_dict


def load_episode(env, root_dir, split, episode, datas, ann_id):
    """Load episode and process datas

    Args:
        root_dir: a string of the root directory of the dataset
        split: a string of the split of the dataset
        episode: a string of the episode name
        datas: a dict of the datas to be saved/loaded
            - static_pcd: a list of nd.arrays with shape (height, width, 3)
            - static_rgb: a list of nd.arrays with shape (height, width, 3)
            - gripper_pcd: a list of nd.arrays with shape (height, width, 3)
            - gripper_rgb: a list of nd.arrays with shape (height, width, 3)
            - proprios: a list of nd.arrays with shape (8,)
            - annotation_id: a list of ints
    """
    data = np.load(f'{root_dir}/{split}/{episode}')

    rgb_static = data['rgb_static']  # (200, 200, 3)
    rgb_gripper = data['rgb_gripper']  # (84, 84, 3)
    depth_static = data['depth_static']  # (200, 200)
    depth_gripper = data['depth_gripper']  # (84, 84)

    # data['robot_obs'] is (15,), data['scene_obs'] is (24,)
    env.reset(robot_obs=data['robot_obs'], scene_obs=data['scene_obs'])
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

    # map RGB to [-1, 1]
    rgb_static = rgb_static / 255. * 2 - 1
    rgb_gripper = rgb_gripper / 255. * 2 - 1

    # Map gripper openess to [0, 1]
    proprio = np.concatenate([
        data['robot_obs'][:3],
        data['robot_obs'][3:6],
        (data['robot_obs'][[-1]] > 0).astype(np.float32)
    ], axis=-1)

    # Put them into a dict
    datas['static_pcd'].append(static_pcd)  # (200, 200, 3)
    datas['static_rgb'].append(rgb_static)  # (200, 200, 3)
    datas['gripper_pcd'].append(gripper_pcd)  # (84, 84, 3)
    datas['gripper_rgb'].append(rgb_gripper)  # (84, 84, 3)
    datas['proprios'].append(proprio)  # (8,)
    datas['annotation_id'].append(ann_id)  # int


def init_datas():
    datas = {
        'static_pcd': [],
        'static_rgb': [],
        'gripper_pcd': [],
        'gripper_rgb': [],
        'proprios': [],
        'annotation_id': []
    }
    return datas


def main(split, args):
    """
    CALVIN contains long videos of "tasks" executed in order
    with noisy transitions between them. The 'annotations' json contains
    info on how to segment those videos.

    Original CALVIN annotations:
    {
        'info': {
            'episodes': [],
            'indx': [(788072, 788136), (899273, 899337), (1427083, 1427147)]
                list of tuples indicating start-end of a task
        },
        'language': {
            'ann': list of str with len=17870, instructions,
            'task': list of str with len=17870, task names,
            'emb': array (17870, 1, 384)
        }
    }

    Save:
    state_dict = [
        frame_ids,  # [0, 1, 2...]
        rgb_pcd,  # tensor [len(frame_ids), ncam, 2, 3, 200, 200]
        action_tensors,  # [tensor(1, 8)]
        camera_dicts,  # [{'front': (0, 0), 'wrist': (0, 0)}]
        gripper_tensors,  # [tensor(1, 8)]
        trajectories,  # [tensor(N, 8) or tensor(2, 8) if keyposes]
        datas['annotation_id']  # [int]
    ]
    """
    annotations = np.load(
        f'{args.root_dir}/{split}/lang_annotations/auto_lang_ann.npy',
        allow_pickle=True
    ).item()
    env = make_env(args.root_dir, split)

    for anno_ind, (start_id, end_id) in enumerate(annotations['info']['indx']):
        # Step 1. load episodes of the same task
        len_anno = len(annotations['info']['indx'])
        if args.tasks is not None and annotations['language']['task'][anno_ind] not in args.tasks:
            continue
        print(f'Processing {anno_ind}/{len_anno}, start_id:{start_id}, end_id:{end_id}')
        datas = init_datas()
        for ep_id in range(start_id, end_id + 1):
            episode = 'episode_{:07d}.npz'.format(ep_id)
            load_episode(
                env,
                args.root_dir,
                split,
                episode,
                datas,
                anno_ind
            )

        # Step 2. detect keyframes within the episode
        _, keyframe_inds = keypoint_discovery(datas['proprios'])

        state_dict = process_datas(
            datas, args.mode, args.traj_len, args.execute_every, keyframe_inds
        )

        # Step 3. determine scene
        if split == 'training':
            scene_info = np.load(
                f'{args.root_dir}/training/scene_info.npy',
                allow_pickle=True
            ).item()
            if ("calvin_scene_B" in scene_info and
                start_id <= scene_info["calvin_scene_B"][1]):
                scene = "B"
            elif ("calvin_scene_C" in scene_info and
                  start_id <= scene_info["calvin_scene_C"][1]):
                scene = "C"
            elif ("calvin_scene_A" in scene_info and
                  start_id <= scene_info["calvin_scene_A"][1]):
                scene = "A"
            else:
                scene = "D"
        else:
            scene = 'D'

        # Step 4. save to .dat file
        ep_save_path = f'{args.save_path}/{split}/{scene}+0/ann_{anno_ind}.dat'
        os.makedirs(os.path.dirname(ep_save_path), exist_ok=True)
        with open(ep_save_path, "wb") as f:
            f.write(blosc.compress(pickle.dumps(state_dict)))

    env.close()


if __name__ == "__main__":
    args = Arguments().parse_args()
    main(args.split, args)
