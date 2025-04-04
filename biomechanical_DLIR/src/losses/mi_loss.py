"""
Wraps MONAI's GlobalMutualInformationLoss to provide a consistent interface for 
both loss computation and metric reporting.

Key Features:
    - `loss(inputs, ddf)`: differentiable MI loss for training
    - `metric(inputs, ddf)`: logs mutual information value for evaluation
    - Warps moving image using MONAI's `Warp` layer
    - Supports global intensity relationships between modalities

Notes:
    - Inputs should include `"fixed"` and `"moving"` image tensors.
    - The deformation field `ddf` is assumed to be in the format (B, 2, H, W) or (B, 3, D, H, W).
"""


from monai.losses import GlobalMutualInformationLoss
from monai.networks.blocks import Warp


class MI:
    def __init__(self,_, **kwargs):
        self.loss_func = GlobalMutualInformationLoss(**kwargs)
        self.warp_layer = Warp()
        self.device = None

    def set_device(self,curr_dev):
        if self.device is None:
            self.device = curr_dev
            self.warp_layer.to(self.device)

    def metric(self, inputs,ddf):
        self.set_device(ddf.device)

        y_true = inputs["fixed"].to(self.device)
        y_moving = inputs["moving"].to(self.device)
        y_pred = self.warp_layer(y_moving, ddf)
        value = self.loss_func(y_pred,y_true).item()
        return {"MI":value}
        
    def loss(self, inputs,ddf):
        self.set_device(ddf.device)
        y_true = inputs["fixed"].to(self.device)
        y_moving = inputs["moving"].to(self.device)

        y_pred = self.warp_layer(y_moving, ddf)
        return self.loss_func(y_pred,y_true)