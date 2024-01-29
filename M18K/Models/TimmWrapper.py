import timm
import torch


class TimmBackboneWrapper(torch.nn.Module):
    def __init__(self, timm_model_name, pretrained=True, in_chans=4):
        super().__init__()
        self.model = timm.create_model(timm_model_name, pretrained=pretrained, in_chans=in_chans, features_only=True,
                                       out_indices=[-1])
        self.out_channels = self.model.feature_info.channels()[0]

    def forward(self, x):
        output = self.model(x)
        return {f"{k}": v for k, v in enumerate(output)}
