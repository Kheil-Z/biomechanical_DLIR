"""
Loss module for regularizing deformations using bending energy.

Wraps MONAI's BendingEnergyLoss to provide a consistent interface for
both loss computation and metric reporting.

Key Features:
    - `loss(inputs, ddf)`: Computes the bending energy of the displacement field `ddf`.
    - `metric(inputs, ddf)`: Returns the scalar bending energy as a logging-friendly metric.
    - Supports both 2D and 3D deformation fields via MONAI’s BendingEnergyLoss.

Notes:
    - The deformation field `ddf` is assumed to be in the format (B, 2, H, W) or (B, 3, D, H, W).
"""

from monai.losses import BendingEnergyLoss


class Bending:
    def __init__(self, _, **kwargs):
        self.loss_func = BendingEnergyLoss(**kwargs)

    def metric(self, inputs, ddf):
        """
        Returns interpretable metrics for evaluation and logging.

        These are not used for optimization.
        """
        metrics = {"bending_energy": self.loss_func(ddf).item()}
        return metrics

    def loss(self, inputs, ddf):
        return self.loss_func(ddf)
