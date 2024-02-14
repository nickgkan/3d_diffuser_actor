# Adapted from https://github.com/pytorch/vision/blob/v0.11.0/torchvision/models/resnet.py

import torch
from torchvision import transforms
from typing import Type, Union, List, Any
from torchvision.models.resnet import _resnet, BasicBlock, Bottleneck, ResNet


def load_resnet50(pretrained: bool = False):
    backbone = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained=pretrained, progress=True)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return backbone, normalize

def load_resnet18(pretrained: bool = False):
    backbone = _resnet('resnet18', Bottleneck, [2, 2, 2, 2], pretrained=pretrained, progress=True)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return backbone, normalize

def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNetFeatures(block, layers, **kwargs)
    if pretrained:
        if int(torch.__version__[0]) <= 1:
            from torch.hub import load_state_dict_from_url
            from torchvision.models.resnet import model_urls
            state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
            model.load_state_dict(state_dict)
        else:
            raise NotImplementedError("Pretrained models not supported in PyTorch 2.0+")
    return model


class ResNetFeatures(ResNet):
    def __init__(self, block, layers, **kwargs):
        super().__init__(block, layers, **kwargs)

    def _forward_impl(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return {
            "res1": x0,
            "res2": x1,
            "res3": x2,
            "res4": x3,
            "res5": x4,
        }
