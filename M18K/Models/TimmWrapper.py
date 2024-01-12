import timm
import torch
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

class TimmBackboneWrapper(torch.nn.Module):
    def __init__(self, timm_model_name, pretrained=True, in_chans=4):
        super().__init__()
        self.model = timm.create_model(timm_model_name, pretrained=pretrained, in_chans=in_chans, features_only=True, out_indices=[-1])
        self.out_channels = self.model.feature_info.channels()[-1]  # Set out_channels to the channels of the last feature layer

    def forward(self, x):
        return self.model(x)

