import logging

import transformers
import torch
import numpy as np

# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel
from diffuser_actor.trajectory_optimization.diffuser_actor import DiffuserActor
from online_evaluation_calvin.evaluate_utils import convert_action
from utils.utils_with_calvin import relative_to_absolute


logger = logging.getLogger(__name__)


def create_model(args, pretrained=True):
    model = DiffusionModel(args)
    if pretrained:
        model.load_pretrained_weights()
    return model


class DiffusionModel(CalvinBaseModel):
    """A wrapper for the DiffuserActor model, which handles
            1. Model initialization
            2. Encodings of instructions
            3. Model inference
            4. Action post-processing
                - quaternion to Euler angles
                - relative to absolute action
    """
    def __init__(self, args):
        self.args = args
        self.policy = self.get_policy()
        self.text_tokenizer, self.text_model = self.get_text_encoder()
        self.reset()

    def get_policy(self):
        """Initialize the model."""
        # Initialize model with arguments
        _model = DiffuserActor(
            backbone=self.args.backbone,
            image_size=tuple(int(x) for x in self.args.image_size.split(",")),
            embedding_dim=self.args.embedding_dim,
            num_vis_ins_attn_layers=self.args.num_vis_ins_attn_layers,
            use_instruction=bool(self.args.use_instruction),
            fps_subsampling_factor=self.args.fps_subsampling_factor,
            gripper_loc_bounds=self.args.gripper_loc_bounds,
            rotation_parametrization=self.args.rotation_parametrization,
            quaternion_format=self.args.quaternion_format,
            diffusion_timesteps=self.args.diffusion_timesteps,
            nhist=self.args.num_history,
            relative=bool(self.args.relative_action),
            lang_enhanced=bool(self.args.lang_enhanced),
        )

        return _model

    def get_text_encoder(self):
        def load_model(encoder) -> transformers.PreTrainedModel:
            if encoder == "bert":
                model = transformers.BertModel.from_pretrained("bert-base-uncased")
            elif encoder == "clip":
                model = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            else:
                raise ValueError(f"Unexpected encoder {encoder}")
            if not isinstance(model, transformers.PreTrainedModel):
                raise ValueError(f"Unexpected encoder {encoder}")
            return model


        def load_tokenizer(encoder) -> transformers.PreTrainedTokenizer:
            if encoder == "bert":
                tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
            elif encoder == "clip":
                tokenizer = transformers.CLIPTokenizer.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
            else:
                raise ValueError(f"Unexpected encoder {encoder}")
            if not isinstance(tokenizer, transformers.PreTrainedTokenizer):
                raise ValueError(f"Unexpected encoder {encoder}")
            return tokenizer


        tokenizer = load_tokenizer(self.args.text_encoder)
        tokenizer.model_max_length = self.args.text_max_length

        model = load_model(self.args.text_encoder)
    
        return tokenizer, model

    def reset(self):
        """Set model to evaluation mode.
        """
        device = self.args.device
        self.policy.eval()
        self.text_model.eval()

        self.policy = self.policy.to(device)
        self.text_model = self.text_model.to(device)

    def load_pretrained_weights(self, state_dict=None):
        if state_dict is None:
            state_dict = torch.load(self.args.checkpoint, map_location="cpu")["weight"]
        model_weights = {}
        for key in state_dict:
            _key = key[7:]
            model_weights[_key] = state_dict[key]
        print(f'Loading weights from {self.args.checkpoint}')
        self.policy.load_state_dict(model_weights)

    def encode_instruction(self, instruction, device="cuda"):
        """Encode string instruction to latent embeddings.

        Args:
            instruction: a string of instruction
            device: a string of device
        
        Returns:
            pred: a tensor of latent embeddings of shape (text_max_length, 512)
        """
        instr = instruction + '.'
        tokens = self.text_tokenizer(instr, padding="max_length")["input_ids"]

        tokens = torch.tensor(tokens).to(device)
        tokens = tokens.view(1, -1)
        with torch.no_grad():
            pred = self.text_model(tokens).last_hidden_state

        return pred

    def step(self, obs, instruction):
        """
        Args:
            obs: a dictionary of observations
                - rgb_obs: a dictionary of RGB images
                - depth_obs: a dictionary of depth images
                - robot_obs: a dictionary of proprioceptive states
            lang_annotation: a string indicates the instruction of the task

        Returns:
            action: predicted action
        """
        device = self.args.device

        # Organize inputs
        trajectory_mask = torch.full(
            [1, self.args.interpolation_length - 1], False
        ).to(device)
        fake_trajectory = torch.full(
            [1, self.args.interpolation_length - 1, self.args.action_dim], 0
        ).to(device)
        rgbs = np.stack([
            obs["rgb_obs"]["rgb_static"], obs["rgb_obs"]["rgb_gripper"]
        ], axis=0).transpose(0, 3, 1, 2) # [ncam, 3, H, W]
        pcds = np.stack([
            obs["pcd_obs"]["pcd_static"], obs["pcd_obs"]["pcd_gripper"]
        ], axis=0).transpose(0, 3, 1, 2) # [ncam, 3, H, W]

        rgbs = torch.as_tensor(rgbs).to(device).unsqueeze(0)
        pcds = torch.as_tensor(pcds).to(device).unsqueeze(0)

        # Crop the images.  See Line 165-166 in datasets/dataset_calvin.py
        rgbs = rgbs[..., 20:180, 20:180]
        pcds = pcds[..., 20:180, 20:180]

        # history of actions
        gripper = torch.as_tensor(obs["proprio"]).to(device).unsqueeze(0)
        gripper = gripper[:, -self.args.num_history:]

        trajectory = self.policy(
            fake_trajectory.float(),
            trajectory_mask,
            rgbs.float(),
            pcds.float(),
            instruction.float(),
            curr_gripper=gripper[..., :7].float(),
            run_inference=True
        )

        # Convert quaternion to Euler angles
        trajectory = convert_action(trajectory)

        if self.args.relative_action:
            # Convert quaternion to Euler angles
            gripper = convert_action(gripper[:, [-1], :])
            # Convert relative action to absolute action
            trajectory = relative_to_absolute(trajectory, gripper)

        # Bound final action by CALVIN statistics
        if self.args.calvin_gripper_loc_bounds is not None:
            trajectory[:, :, :3] = np.clip(
                trajectory[:, :, :3],
                a_min=self.args.calvin_gripper_loc_bounds[0].reshape(1, 1, 3),
                a_max=self.args.calvin_gripper_loc_bounds[1].reshape(1, 1, 3)
            )

        return trajectory
