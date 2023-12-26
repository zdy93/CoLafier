from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
from torchvision.models import resnet50, vgg19_bn
import torch.nn.functional as F
from torchvision.models.resnet import _resnet, ResNet, BasicBlock, Bottleneck, ResNet50_Weights
from torchvision.models.vgg import VGG, cfgs, make_layers, VGG19_BN_Weights
from torchvision.models._api import WeightsEnum, register_model
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.utils import _log_api_usage_once
import torch
import torch.nn as nn
import math


class SinCosLabelEncoding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        assert hidden_size % 2 == 0, f"Cannot use sin/cos label embeddings with odd dim (got dim={hidden_size})"

        self.register_buffer("label_embedding_holder",
                             torch.zeros(hidden_size))
        self.register_buffer("even_div_term", torch.exp(
            2 * torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size)))
        self.register_buffer("odd_div_term", torch.exp(
            2 * torch.arange(1, hidden_size, 2) * -(math.log(10000.0) / hidden_size)))

    def forward(self, label):
        label_shape = label.shape
        hidden_size = self.label_embedding_holder.shape[-1]
        label_embedding_holder = self.label_embedding_holder
        label_embedding_holder = label_embedding_holder.expand(label_shape[0], -1).clone()
        label_expand = label.view(label_shape + (1,)).expand(label_shape + (hidden_size // 2,))
        label_embedding_holder[:, 0::2] = torch.sin(label_expand * self.even_div_term)
        label_embedding_holder[:, 1::2] = torch.cos(label_expand * self.odd_div_term)
        return label_embedding_holder


class ResNetWithlabel(ResNet):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group,
                         replace_stride_with_dilation, norm_layer)
        self.label_embedding = nn.Embedding(num_classes, 512 * block.expansion)
        self.sin_cos_label_embedding = SinCosLabelEncoding(512 * block.expansion)
        self.merge_fc = nn.Linear(2 * 512 * block.expansion, 512 * block.expansion)
        self.layerNorm = nn.LayerNorm(512 * block.expansion)

    def _forward_impl(self, x: Tensor, label: Tensor, get_feat=True, merge_concat=False, activation=None, label_2=None,
                      lam=None, embedding_type="regular") -> Any:
        # See note [TorchScript super()]
        if embedding_type in ["regular", "regular_layernorm"]:
            embedding_layer = self.label_embedding
        else:
            embedding_layer = self.sin_cos_label_embedding
        label_embeds = embedding_layer(label)
        if label_2 is not None:
            label_2_embeds = embedding_layer(label_2)
            label_embeds = lam * label_embeds + (1 - lam) * label_2_embeds
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if merge_concat:
            feature = torch.cat([x, label_embeds], dim=1)
            feature = self.merge_fc(feature)
        else:
            if embedding_type == "regular_layernorm":
                x = self.layerNorm(x)
                label_embeds = self.layerNorm(label_embeds)
            feature = x + label_embeds
        if activation is not None:
            if activation == 'gelu':
                feature = F.gelu(feature)
            elif activation == 'relu':
                feature = F.relu(feature)
        feature = self.layerNorm(feature)
        logits = self.fc(feature)
        if get_feat:
            return logits, feature
        else:
            return logits

    def forward(self, x: Tensor, label: Tensor, get_feat=True, merge_concat=False, activation=None, label_2=None,
                lam=None, embedding_type="regular") -> Any:
        return self._forward_impl(x, label, get_feat, merge_concat, activation, label_2, lam, embedding_type)


def _resnetwithlabel(
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        weights: Optional[WeightsEnum],
        progress: bool,
        **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNetWithlabel(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress), strict=False)

    return model


@register_model()
@handle_legacy_interface(weights=("pretrained", ResNet50_Weights.IMAGENET1K_V1))
def resnet50withlabel(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet50_Weights
        :members:
    """
    weights = ResNet50_Weights.verify(weights)

    return _resnetwithlabel(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


class VGGWithLabel(nn.Module):
    def __init__(
            self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.before_classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Linear(4096, num_classes)
        self.label_embedding = nn.Embedding(num_classes, 4096)
        self.sin_cos_label_embedding = SinCosLabelEncoding(4096)
        self.merge_fc = nn.Linear(2 * 4096, 4096)
        self.layerNorm = nn.LayerNorm(4096)

    def forward(self, x: torch.Tensor, label: Tensor, get_feat=True, merge_concat=False, activation=None, label_2=None,
                lam=None, embedding_type="regular") -> Any:
        if embedding_type == "regular":
            embedding_layer = self.label_embedding
        else:
            embedding_layer = self.sin_cos_label_embedding
        label_embeds = embedding_layer(label)
        if label_2 is not None:
            label_2_embeds = embedding_layer(label_2)
            label_embeds = lam * label_embeds + (1 - lam) * label_2_embeds
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.before_classifier(x)
        if merge_concat:
            feature = torch.cat([x, label_embeds], dim=1)
            feature = self.merge_fc(feature)
        else:
            feature = x + label_embeds
        if activation is not None:
            if activation == 'gelu':
                feature = F.gelu(feature)
            elif activation == 'relu':
                feature = F.relu(feature)
        feature = self.layerNorm(feature)
        x = self.classifier(feature)
        if get_feat:
            return x, feature
        else:
            return x


def _vggwithlabel(cfg: str, batch_norm: bool, weights: Optional[WeightsEnum], progress: bool, **kwargs: Any) -> VGG:
    if weights is not None:
        kwargs["init_weights"] = False
        if weights.meta["categories"] is not None:
            _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = VGGWithLabel(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress), strict=False)
    return model


@register_model()
@handle_legacy_interface(weights=("pretrained", VGG19_BN_Weights.IMAGENET1K_V1))
def vgg19_bn_with_label(*, weights: Optional[VGG19_BN_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-19_BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG19_BN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG19_BN_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG19_BN_Weights
        :members:
    """
    weights = VGG19_BN_Weights.verify(weights)

    return _vggwithlabel("E", True, weights, progress, **kwargs)
