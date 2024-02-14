from typing import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork

from .multihead_custom_attention import MultiheadCustomAttention
from .multihead_flash_attention import MultiheadFlashAttention


def convert_multihead_flash_attention(module, process_group=None):
    r"""Helper function to convert all `MultiheadCustomAttention` layers in
    the model to `MultiheadFlashAttention` layers.

    Follow the implementation of torch.nn.SyncBatchNorm.convert_sync_batchnorm

    Args:
        module (nn.Module): module containing one or more
            `MultiheadCustomAttention` layers
        process_group (optional): process group to scope synchronization,
            default is the whole world

    Returns:
        The original `module` with the converted `MultiheadFlashAttention`
        layers. If the original `module` is a `MultiheadCustomAttention`
        layer, a new `MultiheadFlashAttention` layer object will be returned
        instead.
    """
    module_output = module
    if isinstance(module, MultiheadCustomAttention):
        module_output = MultiheadFlashAttention(
            embed_dim=module.embed_dim,
            num_heads=module.num_heads,
            dropout=module.dropout,
            bias=module.in_proj_bias is not None,
            add_bias_kv=module.bias_k is not None,
            add_zero_attn=module.add_zero_attn,
            kdim=module.kdim,
            vdim=module.vdim,
            slot_competition=module.slot_competition,
            return_kv=module.return_kv,
            gate_attn=module.gate_attn is not None
        )
    for name, child in module.named_children():
        module_output.add_module(
            name, convert_multihead_flash_attention(child, process_group)
        )
    del module
    return module_output


def convert_diffusion_scheduler(model, diffusion_scheduler, **kwargs):
    """Convert model.rotation_noise_scheduler and model.position_noise_scheduler
    to specified scheduler
    """

    config = {}
    if diffusion_scheduler == 'DDIM':
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        diffusion_class = DDIMScheduler
    else:
        raise NotImplementedError

    num_eval_steps = kwargs["num_eval_timesteps"]
    model.rotation_noise_scheduler = diffusion_class(
        num_train_timesteps=model.rotation_noise_scheduler.config.num_train_timesteps,
        beta_schedule=model.rotation_noise_scheduler.config.beta_schedule,
        prediction_type=model.rotation_noise_scheduler.config.prediction_type,
        **config,
    )
    model.position_noise_scheduler = diffusion_class(
        num_train_timesteps=model.position_noise_scheduler.config.num_train_timesteps,
        beta_schedule=model.position_noise_scheduler.config.beta_schedule,
        prediction_type=model.position_noise_scheduler.config.prediction_type,
        **config,
    )
    model.n_steps = kwargs["num_eval_timesteps"]


class EfficientFeaturePyramidNetwork(FeaturePyramidNetwork):
    def __init__(
        self,
        in_channels_list,
        out_channels,
        extra_blocks=None,
        norm_layer=None,
        output_level="res3"
    ):
        super().__init__(
            in_channels_list,
            out_channels,
            extra_blocks,
            norm_layer,
        )
        self.output_level = output_level

    def forward(self, x):
        """
        Computes the FPN for a set of feature maps.

        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.
            level_name: the level name to stop the FPN computation at. If None,
                the entire FPN is computed.

        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from the highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

            # Don't go over all levels to save compute
            if names[idx] == self.output_level:
                names = names[idx:]
                break

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


def convert_efficient_fpn(model):

    def _convert_efficient_fpn(module, output_level):
        module_output = module
        if isinstance(module, FeaturePyramidNetwork):
            in_channels_list = [
                m[0].in_channels 
                for m in module.inner_blocks
            ]
            out_channels = module.inner_blocks[-1][-1].out_channels
            module_output = EfficientFeaturePyramidNetwork(
                in_channels_list, out_channels, output_level=output_level
            )
        for name, child in module.named_children():
            module_output.add_module(
                name, _convert_efficient_fpn(child, output_level)
            )
        del module
        return module_output

    # Very hackish, requires to know the inner structure of model
    output_level = model.prediction_head.feature_map_pyramid[
        model.prediction_head.feat_scales-1
    ]
    return _convert_efficient_fpn(model, output_level)


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, (nn.MultiheadAttention, MultiheadCustomAttention, MultiheadFlashAttention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

def convert_half_precision(model):
    convert_weights(model)