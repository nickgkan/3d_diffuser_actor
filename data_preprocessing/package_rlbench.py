import random
import itertools
from typing import Tuple, Dict, List
import pickle
from pathlib import Path
import json

import blosc
from tqdm import tqdm
import tap
import torch
import numpy as np
import einops
from rlbench.demo import Demo

from utils.utils_with_rlbench import (
    RLBenchEnv,
    keypoint_discovery,
    obs_to_attn,
    transform,
)


class Arguments(tap.Tap):
    data_dir: Path = Path(__file__).parent / "c2farm"
    seed: int = 2
    tasks: Tuple[str, ...] = ("stack_wine",)
    cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist", "front")
    image_size: str = "256,256"
    output: Path = Path(__file__).parent / "datasets"
    max_variations: int = 199
    offset: int = 0
    num_workers: int = 0
    store_intermediate_actions: int = 1


def get_attn_indices_from_demo(
    task_str: str, demo: Demo, cameras: Tuple[str, ...]
) -> List[Dict[str, Tuple[int, int]]]:
    frames = keypoint_discovery(demo)

    frames.insert(0, 0)
    return [{cam: obs_to_attn(demo[f], cam) for cam in cameras} for f in frames]


def get_observation(task_str: str, variation: int,
                    episode: int, env: RLBenchEnv,
                    store_intermediate_actions: bool):
    demos = env.get_demo(task_str, variation, episode)
    demo = demos[0]

    key_frame = keypoint_discovery(demo)
    key_frame.insert(0, 0)

    keyframe_state_ls = []
    keyframe_action_ls = []
    intermediate_action_ls = []

    for i in range(len(key_frame)):
        state, action = env.get_obs_action(demo._observations[key_frame[i]]);
        state = transform(state)
        keyframe_state_ls.append(state.unsqueeze(0))
        keyframe_action_ls.append(action.unsqueeze(0))

        if store_intermediate_actions and i < len(key_frame) - 1:
            intermediate_actions = []
            for j in range(key_frame[i], key_frame[i + 1] + 1):
                _, action = env.get_obs_action(demo._observations[j])
                intermediate_actions.append(action.unsqueeze(0))
            intermediate_action_ls.append(torch.cat(intermediate_actions))

    return demo, keyframe_state_ls, keyframe_action_ls, intermediate_action_ls


class Dataset(torch.utils.data.Dataset):

    def __init__(self, args: Arguments):
        # load RLBench environment
        self.env = RLBenchEnv(
            data_path=args.data_dir,
            image_size=[int(x) for x in args.image_size.split(",")],
            apply_rgb=True,
            apply_pc=True,
            apply_cameras=args.cameras,
        )

        tasks = args.tasks
        variations = range(args.offset, args.max_variations)
        self.items = []
        for task_str, variation in itertools.product(tasks, variations):
            episodes_dir = args.data_dir / task_str / f"variation{variation}" / "episodes"
            episodes = [
                (task_str, variation, int(ep.stem[7:]))
                for ep in episodes_dir.glob("episode*")
            ]
            self.items += episodes

        self.num_items = len(self.items)

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, index: int) -> None:
        task, variation, episode = self.items[index]
        taskvar_dir = args.output / f"{task}+{variation}"
        taskvar_dir.mkdir(parents=True, exist_ok=True)

        (demo,
         keyframe_state_ls,
         keyframe_action_ls,
         intermediate_action_ls) = get_observation(
            task, variation, episode, self.env,
            bool(args.store_intermediate_actions)
        )

        state_ls = einops.rearrange(
            keyframe_state_ls,
            "t 1 (m n ch) h w -> t n m ch h w",
            ch=3,
            n=len(args.cameras),
            m=2,
        )

        frame_ids = list(range(len(state_ls) - 1))
        num_frames = len(frame_ids)
        attn_indices = get_attn_indices_from_demo(task, demo, args.cameras)

        state_dict: List = [[] for _ in range(6)]
        print("Demo {}".format(episode))
        state_dict[0].extend(frame_ids)
        state_dict[1] = state_ls[:-1].numpy()
        state_dict[2].extend(keyframe_action_ls[1:])
        state_dict[3].extend(attn_indices)
        state_dict[4].extend(keyframe_action_ls[:-1])  # gripper pos
        state_dict[5].extend(intermediate_action_ls)   # traj from gripper pos to keyframe action

        with open(taskvar_dir / f"ep{episode}.dat", "wb") as f:
            f.write(blosc.compress(pickle.dumps(state_dict)))


if __name__ == "__main__":
    args = Arguments().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset = Dataset(args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,
    )

    for _ in tqdm(dataloader):
        continue
