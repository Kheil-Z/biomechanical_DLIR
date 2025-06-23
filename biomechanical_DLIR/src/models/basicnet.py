"""
basic_net.py

Additional backbones for VoxelMorph style registration

This module defines the BasicNet class, a wrapper for MONAI-based registration models.
It supports several 3D registration backbones (UNet, SegResNet, AttentionUnet) and integrates
them into a unified interface compatible with our training framework.

BasicNet expects a dictionary with 'fixed' and 'moving' images as input and outputs a dense
displacement field (DDF). If specified, it applies a stationary velocity field integration
(DVF2DDF) to ensure diffeomorphic transformations.

Classes:
    - BasicNet: Main registration model wrapper with customizable backbone and integration.

Dependencies:
    - MONAI (network backbones, warping and integration blocks)
    - PyTorch

Usage:
    model = BasicNet(params, device)
    ddf = model({"fixed": fixed_img, "moving": moving_img})
"""

import torch
from monai.networks.blocks import DVF2DDF, Warp
from monai.networks.nets import AttentionUnet  # UNETR, BasicUNet, RegUNet
from monai.networks.nets import SegResNet, UNet
from src.models import BaseModel


class BasicNet(BaseModel):
    def __init__(self, params, device):
        self.params = params
        super().__init__(params, device)

        self.integration_steps = params["integration_steps"]
        self.make_backbone()
        self.backbone.to(device)
        # self.set_booleans()
        self.warp_layer = Warp().to(device)

    # TODO Kwargs....
    def make_backbone(self):
        """
        Initialize the registration backbone according to the parameters.

        Supported backbones:
            - "SegResNet"
            - "UNet"
            - "attention" (i.e., AttentionUnet)

        Raises:
            ValueError: If the specified backbone is not recognized.
        """
        backbone_class = self.params["model_kwargs"]["backbone"]
        # kwargs = copy.deepcopy(self.params["model_kwargs"])
        # del kwargs["backbone"]
        channels = (16, 32, 64, 128, 256, 512)
        strides = (2, 2, 2, 2, 2, 2)
        if "channels" in self.params["model_kwargs"]:
            channels = self.params["model_kwargs"]["channels"]
            strides = self.params["model_kwargs"]["strides"]
        if backbone_class == "SegResNet":
            model = SegResNet(
                spatial_dims=3,
                in_channels=2,
                out_channels=3,
                blocks_down=[1, 2, 2, 4],
                blocks_up=[1, 1, 1],
                init_filters=16,
                dropout_prob=0.0,
            )
        elif backbone_class == "UNet":
            model = UNet(
                spatial_dims=3,
                in_channels=2,
                out_channels=3,
                channels=channels,
                strides=strides,
                num_res_units=2,
            )
        elif backbone_class == "attention":
            model = AttentionUnet(
                spatial_dims=3,
                in_channels=2,
                out_channels=3,
                channels=channels,
                strides=strides,
                kernel_size=3,
                up_kernel_size=3,
                dropout=0.0,
            )
        else:
            raise ValueError(f'Unrecognized Backbone value "{backbone_class}"')
        self.backbone = model

    def forward(self, data):
        """
        Forward pass through the model.

        Args:
            data (dict): Dictionary with keys "fixed" and "moving", each a 3D image tensor.

        Returns:
            torch.Tensor: The dense displacement field (DDF).
        """
        # Extract Fixed and moving images
        fixed = data["fixed"].to(self.device)
        moving = data["moving"].to(self.device)

        # Concatenate for input:
        input = torch.cat(([fixed, moving]), dim=1)

        # Forward in backbone
        flow = self.backbone(input)

        # Integrate flow if necessary
        if self.integration_steps > 0:
            ddf = DVF2DDF(num_steps=self.integration_steps)(flow)
        else:
            ddf = flow
        return ddf
