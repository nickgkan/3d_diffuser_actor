import einops
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork

from diffuser_actor.utils.position_encodings import RotaryPositionEncoding3D
from diffuser_actor.utils.layers import FFWRelativeCrossAttentionModule
from diffuser_actor.utils.utils import (
    normalise_quat,
    sample_ghost_points_uniform_cube,
    sample_ghost_points_uniform_sphere,
    compute_rotation_matrix_from_ortho6d
)
from diffuser_actor.utils.resnet import load_resnet50
from diffuser_actor.utils.clip import load_clip


class Act3D(nn.Module):

    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 num_attn_heads=4,
                 num_ghost_point_cross_attn_layers=2,
                 num_query_cross_attn_layers=2,
                 num_vis_ins_attn_layers=2,
                 rotation_parametrization="quat_from_query",
                 gripper_loc_bounds=None,
                 num_ghost_points=1000,
                 num_ghost_points_val=10000,
                 weight_tying=True,
                 gp_emb_tying=True,
                 ins_pos_emb=False,
                 num_sampling_level=3,
                 fine_sampling_ball_diameter=0.16,
                 regress_position_offset=False,
                 use_instruction=False):
        super().__init__()
        assert backbone in ["resnet", "clip"]
        assert image_size in [(128, 128), (256, 256)]
        assert rotation_parametrization in [
            "quat_from_top_ghost", "quat_from_query",
            "6D_from_top_ghost", "6D_from_query"
        ]
        assert num_sampling_level in [1, 2, 3, 4]

        self.image_size = image_size
        self.rotation_parametrization = rotation_parametrization
        self.num_ghost_points = num_ghost_points // num_sampling_level
        self.num_ghost_points_val = num_ghost_points_val // num_sampling_level
        self.num_sampling_level = num_sampling_level
        self.sampling_ball_diameter_pyramid = [
            None,
            fine_sampling_ball_diameter,
            fine_sampling_ball_diameter / 4.0,
            fine_sampling_ball_diameter / 16.0
        ]
        self.gripper_loc_bounds = np.array(gripper_loc_bounds)
        self.regress_position_offset = regress_position_offset
        self.weight_tying = weight_tying
        self.gp_emb_tying = gp_emb_tying
        self.ins_pos_emb = ins_pos_emb

        # Frozen backbone
        if backbone == "resnet":
            self.backbone, self.normalize = load_resnet50()
        elif backbone == "clip":
            self.backbone, self.normalize = load_clip()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Semantic visual features at different scales
        self.feature_pyramid = FeaturePyramidNetwork(
            [64, 256, 512, 1024, 2048], embedding_dim)
        if self.image_size == (128, 128):
            # Coarse RGB features are the 2nd layer of the feature pyramid at 1/4 resolution (32x32)
            # Fine RGB features are the 1st layer of the feature pyramid at 1/2 resolution (64x64)
            self.coarse_feature_map = ['res2', 'res1', 'res1', 'res1']
            self.downscaling_factor_pyramid = [4, 2, 2, 2]
        elif self.image_size == (256, 256):
            # Coarse RGB features are the 3rd layer of the feature pyramid at 1/8 resolution (32x32)
            # Fine RGB features are the 1st layer of the feature pyramid at 1/2 resolution (128x128)
            self.feature_map_pyramid = ['res3', 'res1', 'res1', 'res1']
            self.downscaling_factor_pyramid = [8, 2, 2, 2]

        # 3D relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # Ghost points learnable initial features
        self.ghost_points_embed_pyramid = nn.ModuleList()
        if self.gp_emb_tying:
            gp_emb = nn.Embedding(1, embedding_dim)
            for _ in range(self.num_sampling_level):
                self.ghost_points_embed_pyramid.append(gp_emb)
        else:
            for _ in range(self.num_sampling_level):
                self.ghost_points_embed_pyramid.append(nn.Embedding(1, embedding_dim))

        # Current gripper learnable features
        self.curr_gripper_embed = nn.Embedding(1, embedding_dim)

        # Query learnable features
        self.query_embed = nn.Embedding(1, embedding_dim)

        # Ghost point cross-attention to visual features and current gripper position
        self.ghost_point_cross_attn_pyramid = nn.ModuleList()
        if self.weight_tying:
            ghost_point_cross_attn = FFWRelativeCrossAttentionModule(
                embedding_dim, num_attn_heads,
                num_ghost_point_cross_attn_layers,
                use_adaln=False)
            for _ in range(self.num_sampling_level):
                self.ghost_point_cross_attn_pyramid.append(ghost_point_cross_attn)
        else:
            for _ in range(self.num_sampling_level):
                ghost_point_cross_attn = FFWRelativeCrossAttentionModule(
                    embedding_dim, num_attn_heads,
                    num_ghost_point_cross_attn_layers,
                    use_adaln=False)
                self.ghost_point_cross_attn_pyramid.append(ghost_point_cross_attn)

        self.use_instruction = use_instruction
        # Visual tokens cross-attention to language instructions
        if self.use_instruction:
            self.vis_ins_attn_pyramid = nn.ModuleList()
            if self.weight_tying:
                vis_ins_cross_attn = FFWRelativeCrossAttentionModule(
                    embedding_dim, num_attn_heads, num_vis_ins_attn_layers,
                    use_adaln=False)
                for i in range(self.num_sampling_level):
                    self.vis_ins_attn_pyramid.append(vis_ins_cross_attn)
            else:
                for i in range(self.num_sampling_level):
                    vis_ins_cross_attn = FFWRelativeCrossAttentionModule(
                        embedding_dim, num_attn_heads, num_vis_ins_attn_layers,
                        use_adaln=False)
                    self.vis_ins_attn_pyramid.append(vis_ins_cross_attn)

        # Query cross-attention to visual features, ghost points, and the current gripper position
        self.query_cross_attn_pyramid = nn.ModuleList()
        if self.weight_tying:
            coarse_query_cross_attn = FFWRelativeCrossAttentionModule(
                embedding_dim, num_attn_heads, num_query_cross_attn_layers,
                use_adaln=False)
            for i in range(self.num_sampling_level):
                self.query_cross_attn_pyramid.append(coarse_query_cross_attn)
        else:
            for i in range(self.num_sampling_level):
                coarse_query_cross_attn = FFWRelativeCrossAttentionModule(
                    embedding_dim, num_attn_heads, num_query_cross_attn_layers,
                    use_adaln=False)
                self.query_cross_attn_pyramid.append(coarse_query_cross_attn)

        # Ghost point offset prediction
        if self.regress_position_offset:
            self.ghost_point_offset_predictor = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 3)
            )

        # Gripper rotation (quaternion) and binary opening prediction
        if "quat" in self.rotation_parametrization:
            self.rotation_dim = 4
        elif "6D" in self.rotation_parametrization:
            self.rotation_dim = 6
        self.gripper_state_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, self.rotation_dim + 1)
        )

        # Instruction encoder
        if self.use_instruction:
            self.instruction_encoder = nn.Linear(512, embedding_dim)
            if self.ins_pos_emb:
                self._num_words = 53
                self.instr_position_embedding = nn.Embedding(self._num_words, embedding_dim)
                self.instr_position_norm = nn.LayerNorm(embedding_dim)

    def forward(self, visible_rgb, visible_pcd, instruction, curr_gripper, gt_action=None):
        """
        Arguments:
            visible_rgb: (batch x history, num_cameras, 3, height, width) in [0, 1]
            visible_pcd: (batch x history, num_cameras, 3, height, width) in world coordinates
            curr_gripper: (batch x history, 8)
            instruction: (batch x history, max_instruction_length, 512)
            gt_action: (batch x history, 8) in world coordinates
        """
        total_timesteps, num_cameras, _, height, width = visible_rgb.shape
        device = visible_rgb.device
        if gt_action is not None:
            gt_position = gt_action[:, :3].unsqueeze(1).detach()
        else:
            gt_position = None
        curr_gripper = curr_gripper[:, :3]

        # Compute visual features at different scales and their positional embeddings
        visible_rgb_features_pyramid, visible_rgb_pos_pyramid, visible_pcd_pyramid = self._compute_visual_features(
            visible_rgb, visible_pcd, num_cameras)

        # Encode instruction
        if self.use_instruction:
            instruction_features = self.instruction_encoder(instruction)

            if self.ins_pos_emb:
                position = torch.arange(self._num_words)
                position = position.unsqueeze(0).to(instruction_features.device)

                pos_emb = self.instr_position_embedding(position)
                pos_emb = self.instr_position_norm(pos_emb)
                pos_emb = einops.repeat(pos_emb, "1 k d -> b k d", b=instruction_features.shape[0])

                instruction_features += pos_emb

            instruction_features = einops.rearrange(instruction_features, "bt l c -> l bt c")
            instruction_dummy_pos = torch.zeros(total_timesteps, instruction_features.shape[0], 3, device=device)
            instruction_dummy_pos = self.relative_pe_layer(instruction_dummy_pos)
        else:
            instruction_features = None
            instruction_dummy_pos = None

        # Compute current gripper position features and positional embeddings
        curr_gripper_pos = self.relative_pe_layer(curr_gripper.unsqueeze(1))
        curr_gripper_features = self.curr_gripper_embed.weight.repeat(total_timesteps, 1).unsqueeze(0)

        ghost_pcd_features_pyramid = []
        ghost_pcd_pyramid = []
        position_pyramid = []
        visible_rgb_mask_pyramid = []
        ghost_pcd_masks_pyramid = []

        for i in range(self.num_sampling_level):
            # Sample ghost points
            if i == 0:
                anchor = None
            else:
                anchor = gt_position if gt_position is not None else position_pyramid[-1]
            ghost_pcd_i = self._sample_ghost_points(total_timesteps, device, level=i, anchor=anchor)

            if i == 0:
                # Coarse RGB features
                visible_rgb_features_i = visible_rgb_features_pyramid[i]
                visible_rgb_pos_i = visible_rgb_pos_pyramid[i]
                ghost_pcd_context_features_i = einops.rearrange(
                    visible_rgb_features_i, "b ncam c h w -> (ncam h w) b c")
            else:
                # Local fine RGB features
                l2_pred_pos = ((position_pyramid[-1] - visible_pcd_pyramid[i]) ** 2).sum(-1).sqrt()
                indices = l2_pred_pos.topk(k=32 * 32 * num_cameras, dim=-1, largest=False).indices

                visible_rgb_features_i = einops.rearrange(
                    visible_rgb_features_pyramid[i], "b ncam c h w -> b (ncam h w) c")
                visible_rgb_features_i = torch.stack([
                    f[i] for (f, i) in zip(visible_rgb_features_i, indices)])
                visible_rgb_pos_i = torch.stack([
                    f[i] for (f, i) in zip(visible_rgb_pos_pyramid[i], indices)])
                ghost_pcd_context_features_i = einops.rearrange(
                    visible_rgb_features_i, "b npts c -> npts b c")

            # Compute ghost point features and their positional embeddings by attending to visual
            # features and current gripper position
            ghost_pcd_context_features_i = torch.cat(
                [ghost_pcd_context_features_i, curr_gripper_features], dim=0)
            ghost_pcd_context_pos_i = torch.cat([visible_rgb_pos_i, curr_gripper_pos], dim=1)
            if self.use_instruction:
                ghost_pcd_context_features_i = self.vis_ins_attn_pyramid[i](
                    query=ghost_pcd_context_features_i, value=instruction_features,
                    query_pos=None, value_pos=None
                )[-1]

                ghost_pcd_context_features_i = torch.cat(
                    [ghost_pcd_context_features_i, instruction_features], dim=0)
                ghost_pcd_context_pos_i = torch.cat(
                    [ghost_pcd_context_pos_i, instruction_dummy_pos], dim=1)
            (
                ghost_pcd_features_i,
                ghost_pcd_pos_i,
                ghost_pcd_to_visible_rgb_attn_i
            ) = self._compute_ghost_point_features(
                ghost_pcd_i, ghost_pcd_context_features_i, ghost_pcd_context_pos_i,
                total_timesteps, level=i
            )

            # Initialize query features
            if i == 0:
                query_features = self.query_embed.weight.unsqueeze(1).repeat(1, total_timesteps, 1)

            query_context_features_i = ghost_pcd_context_features_i
            query_context_pos_i = ghost_pcd_context_pos_i

            if i == 0:
                # Given the query is not localized yet, we don't use positional embeddings
                query_pos_i = None
                context_pos_i = None
            else:
                # Now that the query is localized, we use positional embeddings
                query_pos_i = self.relative_pe_layer(position_pyramid[-1])
                context_pos_i = query_context_pos_i

            # The query cross-attends to context features (visual features and the current gripper position)
            query_features = self._compute_query_features(
                query_features, query_context_features_i,
                query_pos_i, context_pos_i,
                level=i
            )

            # The query decodes a mask over ghost points (used to predict the gripper position) and over visual
            # features (for visualization only)
            ghost_pcd_masks_i, visible_rgb_mask_i = self._decode_mask(
                query_features,
                ghost_pcd_features_i, ghost_pcd_to_visible_rgb_attn_i,
                height, width, level=i
            )
            query_features = query_features[-1]

            top_idx = torch.max(ghost_pcd_masks_i[-1], dim=-1).indices
            ghost_pcd_i = einops.rearrange(ghost_pcd_i, "b npts c -> b c npts")
            position_i = ghost_pcd_i[torch.arange(total_timesteps), :, top_idx].unsqueeze(1)

            ghost_pcd_pyramid.append(ghost_pcd_i)
            ghost_pcd_features_pyramid.append(ghost_pcd_features_i)
            position_pyramid.append(position_i)
            visible_rgb_mask_pyramid.append(visible_rgb_mask_i)
            ghost_pcd_masks_pyramid.append(ghost_pcd_masks_i)

        # Regress an offset from the ghost point's position to the predicted position
        if self.regress_position_offset:
            fine_ghost_pcd_offsets = self.ghost_point_offset_predictor(ghost_pcd_features_i)
            fine_ghost_pcd_offsets = einops.rearrange(fine_ghost_pcd_offsets, "npts b c -> b c npts")
        else:
            fine_ghost_pcd_offsets = None

        ghost_pcd = ghost_pcd_i
        ghost_pcd_masks = ghost_pcd_masks_i
        ghost_pcd_features = ghost_pcd_features_i

        # Predict the next gripper action (position, rotation, gripper opening)
        position, rotation, gripper = self._predict_action(
            ghost_pcd_masks[-1], ghost_pcd, ghost_pcd_features, query_features, total_timesteps,
            fine_ghost_pcd_offsets if self.regress_position_offset else None
        )
        # position = position_pyramid[-1].squeeze(1)

        return {
            # Action
            "position": position,
            "rotation": rotation,
            "gripper": gripper,
            # Auxiliary outputs used to compute the loss or for visualization
            "position_pyramid": position_pyramid,
            "visible_rgb_mask_pyramid": visible_rgb_mask_pyramid,
            "ghost_pcd_masks_pyramid":  ghost_pcd_masks_pyramid,
            "ghost_pcd_pyramid": ghost_pcd_pyramid,
            "fine_ghost_pcd_offsets": fine_ghost_pcd_offsets if self.regress_position_offset else None,
            # Return intermediate results
            "visible_rgb_features_pyramid": visible_rgb_features_pyramid,
            "visible_pcd_pyramid": visible_pcd_pyramid,
            "query_features": query_features,
            "instruction_features": instruction_features,
            "instruction_dummy_pos": instruction_dummy_pos,
        }

    def prepare_action(self, pred) -> torch.Tensor:
        rotation = pred["rotation"]
        return torch.cat(
            [pred["position"], rotation, pred["gripper"]],
            dim=1,
        )

    def _compute_visual_features(self, visible_rgb, visible_pcd, num_cameras):
        """Compute visual features at different scales and their positional embeddings."""
        ncam = visible_rgb.shape[1]

        # Pass each view independently through backbone
        visible_rgb = einops.rearrange(visible_rgb, "bt ncam c h w -> (bt ncam) c h w")
        visible_rgb = self.normalize(visible_rgb)
        visible_rgb_features = self.backbone(visible_rgb)

        # Pass visual features through feature pyramid network
        visible_rgb_features = self.feature_pyramid(visible_rgb_features)

        visible_pcd = einops.rearrange(visible_pcd, "bt ncam c h w -> (bt ncam) c h w")

        visible_rgb_features_pyramid = []
        visible_rgb_pos_pyramid = []
        visible_pcd_pyramid = []

        for i in range(self.num_sampling_level):
            visible_rgb_features_i = visible_rgb_features[self.feature_map_pyramid[i]]
            visible_pcd_i = F.interpolate(
                visible_pcd, scale_factor=1. / self.downscaling_factor_pyramid[i], mode='bilinear')
            h, w = visible_pcd_i.shape[-2:]
            visible_pcd_i = einops.rearrange(
                visible_pcd_i, "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras)
            visible_rgb_pos_i = self.relative_pe_layer(visible_pcd_i)
            visible_rgb_features_i = einops.rearrange(
                visible_rgb_features_i, "(bt ncam) c h w -> bt ncam c h w", ncam=num_cameras)

            visible_rgb_features_pyramid.append(visible_rgb_features_i)
            visible_rgb_pos_pyramid.append(visible_rgb_pos_i)
            visible_pcd_pyramid.append(visible_pcd_i)

        return visible_rgb_features_pyramid, visible_rgb_pos_pyramid, visible_pcd_pyramid

    def _sample_ghost_points(self, total_timesteps, device, level, anchor=None):
        """Sample ghost points.

        If level==0, sample points uniformly within the workspace bounds.

        If level>0, sample points uniformly within a local sphere
        of the workspace bounds centered around the anchor.
        """
        if self.training:
            num_ghost_points = self.num_ghost_points
        else:
            num_ghost_points = self.num_ghost_points_val

        if level == 0:
            bounds = np.stack([self.gripper_loc_bounds for _ in range(total_timesteps)])
            uniform_pcd = np.stack([
                sample_ghost_points_uniform_cube(
                    bounds=bounds[i],
                    num_points=num_ghost_points
                )
                for i in range(total_timesteps)
            ])

        elif level >= 1:
            anchor_ = anchor[:, 0].cpu().numpy()
            bounds_min = np.clip(
                anchor_ - self.sampling_ball_diameter_pyramid[level] / 2,
                a_min=self.gripper_loc_bounds[0], a_max=self.gripper_loc_bounds[1]
            )
            bounds_max = np.clip(
                anchor_ + self.sampling_ball_diameter_pyramid[level] / 2,
                a_min=self.gripper_loc_bounds[0], a_max=self.gripper_loc_bounds[1]
            )
            bounds = np.stack([bounds_min, bounds_max], axis=1)
            uniform_pcd = np.stack([
                sample_ghost_points_uniform_sphere(
                    center=anchor_[i],
                    radius=self.sampling_ball_diameter_pyramid[level] / 2,
                    bounds=bounds[i],
                    num_points=num_ghost_points
                )
                for i in range(total_timesteps)
            ])

        uniform_pcd = torch.from_numpy(uniform_pcd).float().to(device)

        return uniform_pcd

    def _compute_ghost_point_features(self,
                                      ghost_pcd, context_features, context_pos,
                                      total_timesteps, level):
        """
        Ghost points cross-attend to context features (visual features, instruction features,
        and current gripper position).
        """
        embed = self.ghost_points_embed_pyramid[level]
        attn_layers = self.ghost_point_cross_attn_pyramid[level]

        # Initialize ghost point features and positional embeddings
        ghost_pcd_pos = self.relative_pe_layer(ghost_pcd)
        num_ghost_points = ghost_pcd.shape[1]
        ghost_pcd_features = embed.weight.unsqueeze(0).repeat(num_ghost_points, total_timesteps, 1)

        # Ghost points cross-attend to visual features and current gripper position
        ghost_pcd_features = attn_layers(
            query=ghost_pcd_features, value=context_features,
            query_pos=ghost_pcd_pos, value_pos=context_pos
        )[-1]

        ghost_pcd_to_visible_rgb_attn = None

        return ghost_pcd_features, ghost_pcd_pos, ghost_pcd_to_visible_rgb_attn

    def _compute_query_features(self,
                                query_features, context_features,
                                query_pos, context_pos,
                                level):
        """The query cross-attends to context features (visual features, instruction features,
        and current gripper position)."""
        attn_layers = self.query_cross_attn_pyramid[level]

        query_features = attn_layers(
            query=query_features, value=context_features,
            query_pos=query_pos, value_pos=context_pos
        )

        return query_features

    def _decode_mask(self,
                     query_features,
                     ghost_pcd_features, ghost_pcd_to_visible_rgb_attn,
                     rgb_height, rgb_width, level):
        """
        The query decodes a mask over ghost points (used to predict the gripper position) and over visual
        features (for visualization only).
        """
        h = rgb_height // self.downscaling_factor_pyramid[level]
        w = rgb_width // self.downscaling_factor_pyramid[level]

        ghost_pcd_masks = [einops.einsum(f.squeeze(0), ghost_pcd_features, "bt c, npts bt c -> bt npts")
                           for f in query_features]

        # Extract attention from top ghost point to visual features for visualization
        if ghost_pcd_to_visible_rgb_attn is not None:
            top_idx = torch.max(ghost_pcd_masks[-1], dim=-1).indices
            visible_rgb_mask = ghost_pcd_to_visible_rgb_attn[torch.arange(len(top_idx)), top_idx]
            visible_rgb_mask = einops.rearrange(visible_rgb_mask, "bt (ncam h w) -> bt ncam h w", h=h, w=w)
            visible_rgb_mask = F.interpolate(visible_rgb_mask, size=(rgb_height, rgb_width), mode="nearest")
        else:
            visible_rgb_mask = None

        return ghost_pcd_masks, visible_rgb_mask

    def _predict_action(self,
                        ghost_pcd_mask, ghost_pcd, ghost_pcd_features, query_features, total_timesteps,
                        fine_ghost_pcd_offsets=None):
        """Compute the predicted action (position, rotation, opening) from the predicted mask."""
        # Select top-scoring ghost point
        top_idx = torch.max(ghost_pcd_mask, dim=-1).indices
        position = ghost_pcd[torch.arange(total_timesteps), :, top_idx]

        # Add an offset regressed from the ghost point's position to the predicted position
        if fine_ghost_pcd_offsets is not None:
            position = position + fine_ghost_pcd_offsets[torch.arange(total_timesteps), :, top_idx]

        # Predict rotation and gripper opening
        if self.rotation_parametrization in ["quat_from_top_ghost", "6D_from_top_ghost"]:
            ghost_pcd_features = einops.rearrange(ghost_pcd_features, "npts bt c -> bt npts c")
            features = ghost_pcd_features[torch.arange(total_timesteps), top_idx]
        elif self.rotation_parametrization in ["quat_from_query", "6D_from_query"]:
            features = query_features.squeeze(0)

        pred = self.gripper_state_predictor(features)

        if "quat" in self.rotation_parametrization:
            rotation = normalise_quat(pred[:, :self.rotation_dim])
        elif "6D" in self.rotation_parametrization:
            rotation = compute_rotation_matrix_from_ortho6d(pred[:, :self.rotation_dim])

        gripper = torch.sigmoid(pred[:, self.rotation_dim:])

        return position, rotation, gripper
