from torchvision.models.detection.rpn import AnchorGenerator
import torchvision

import timm
import torch
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
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


backbone = TimmBackboneWrapper("resnet50", pretrained=True, in_chans=4)

anchor_generator = AnchorGenerator(
    sizes=(32, 64, 128, 256, 512),
    aspect_ratios=((0.5, 1.0, 2.0),(0.5, 1.0, 2.0),(0.5, 1.0, 2.0),(0.5, 1.0, 2.0),(0.5, 1.0, 2.0),)
)

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)

mask_head = MaskRCNNHeads(256, (256, 256, 256, 256), 1)

mask_predictor = MaskRCNNPredictor(256, 256, 3)

# put the pieces together inside a Faster-RCNN model
model = torchvision.models.detection.MaskRCNN(
    backbone.model,
    num_classes=3,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler,
    image_mean=[0.0, 0.0, 0.0, 0.0],
    image_std=[1.0, 1.0, 1.0, 1.0],
    mask_head=mask_head,
    # mask_predictor= mask_predictor
)

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from M18K.Models.MaskRCNN import MaskRCNN

from M18K.Data.Dataset import M18KDataset

ds = M18KDataset(root="M18K/Data/train", transforms=None, train=True, depth=True)
# model = MaskRCNN(depth=True).model
mm = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
# mm.backbone.model = backbone.model
# mm.transform = model.transform
i, t = ds[0]
print(model([i], [t]))
