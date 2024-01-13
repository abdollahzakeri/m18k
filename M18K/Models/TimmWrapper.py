import timm
import torch
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torch import nn

class TimmBackboneWrapper(torch.nn.Module):
    def __init__(self, timm_model_name, pretrained=True, in_chans=4):
        super().__init__()
        self.backbone = timm.create_model(timm_model_name, pretrained=pretrained, in_chans=in_chans, features_only=True,
                                          out_indices=[1, 2, 3, 4])
        return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        in_channels_list = self.backbone.feature_info.channels()
        self.model = BackboneWithFPN(
            self.backbone,
            return_layers,
            in_channels_list,
            256,
            extra_blocks=LastLevelMaxPool(),
            norm_layer=nn.BatchNorm2d
        )
        self.out_channels = 256  # [64, 256, 512, 1024, 2048]

    def forward(self, x):
        output = self.model(x)
        return {f"{k}": v for k, v in enumerate(output)}
