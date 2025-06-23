"""
Loss module for deformation-based learning focusing on Jacobian determinant regularization.

This class implements a loss term that penalizes undesirable local volume changes in
deformation fields by enforcing regularity of the Jacobian determinant. It promotes
diffeomorphic deformations and discourages foldings by applying a squared log penalty
on the determinant of the Jacobian matrix.

Key Features:
    - `loss(inputs, ddf)`: Computes the log-squared penalty on the positive part of the Jacobian determinant.
    - `metric(inputs, ddf)`: Provides interpretable metrics such as standard deviation of log-determinants
      and percentage of foldings (negative determinants).
    - Handles both 2D and 3D deformations.
    - Internally computes deformation gradients via finite differences.
    - Automatically accounts for u = φ + id when computing gradients.

Notes:
    - Tensor layout is assumed to be (B, 2, H, W) for 2D and (B, 3, D, H, W) for 3D deformation fields.
    - Border slices are replaced with ones to avoid artifacts in gradient computation and determinant evaluation.
    - Negative determinants are monitored through metrics and thresholding.
"""

import torch


class DetJac:
    """
    Class to compute Detemrinant of Jacobian loss
    """

    def __init__(self, params, **kwargs):
        self.params = params

    def _computeDetJac2D(self, phi):
        dx = torch.gradient(phi[:, 0, :, :], axis=(1, 2))  # Warning
        dy = torch.gradient(phi[:, 1, :, :], axis=(1, 2))  # Warning

        phiX_dx, phiX_dy = dx
        phiY_dx, phiY_dy = dy

        # +1 because u = phi + id
        phiX_dx += 1
        phiY_dy += 1
        determinant = phiX_dx * phiY_dy - phiY_dx * phiX_dy
        return determinant

    def _computeDetJac3D(self, phi):
        dx = torch.gradient(phi[:, 0, :, :, :], axis=(1, 2, 3))
        dy = torch.gradient(phi[:, 1, :, :, :], axis=(1, 2, 3))
        dz = torch.gradient(phi[:, 2, :, :, :], axis=(1, 2, 3))

        phiX_dx, phiX_dy, phiX_dz = dx
        phiY_dx, phiY_dy, phiY_dz = dy
        phiZ_dx, phiZ_dy, phiZ_dz = dz

        # +1 because u = phi + id
        phiX_dx += 1
        phiY_dy += 1
        phiZ_dz += 1

        plus = (
            (phiX_dx * phiY_dy * phiZ_dz)
            + (phiX_dy * phiY_dz * phiZ_dx)
            + (phiX_dz * phiY_dx * phiZ_dy)
        )
        minus = (
            (phiX_dz * phiY_dy * phiZ_dx)
            + (phiX_dy * phiY_dx * phiZ_dz)
            + (phiX_dx * phiY_dz * phiZ_dy)
        )
        determinant = plus - minus

        return determinant

    def metric(self, inputs, ddf):
        """
        Returns interpretable metrics for evaluation and logging.

        These are not used for optimization.
        """
        # device = ddf.device

        inp_shape = ddf.shape
        assert (inp_shape[1] == 2 and (len(inp_shape) == 4)) or (
            inp_shape[1] == 3 and (len(inp_shape) == 5)
        ), f" Input shape {inp_shape} incorrect, expected (batch, 2, h, w) for 2D or (batch, 3, d, h, w) for 3D data"

        # Get problem dimension (2d/3d)
        dim = len(inp_shape) - 2

        if dim == 2:
            determinant = self._computeDetJac2D(ddf)
            # Border effects :
            determinant[:, 0, :] = torch.ones_like(determinant[:, 0, :])
            determinant[:, :, 0] = torch.ones_like(determinant[:, :, 0])
        else:
            determinant = self._computeDetJac3D(ddf)
            # Border effects :
            determinant[:, 0, :, :] = torch.ones_like(determinant[:, 0, :, :])
            determinant[:, :, 0, :] = torch.ones_like(determinant[:, :, 0, :])
            determinant[:, :, :, 0] = torch.ones_like(determinant[:, :, :, 0])

        negative_dets = torch.sum(determinant < 0.0).item()
        positive_dets = torch.sum(determinant >= 0.0).item()
        std_log_jac = (
            (torch.nn.Threshold(1e-9, 1e-9)(determinant)).log().std().item()
        )  # determinant[determinant>0].log().std().item()

        metrics = {
            "std_log_jac": std_log_jac,
            "negative_dets": negative_dets,
            "positive_dets": positive_dets,
            "percent_foldings": 100 * negative_dets / (positive_dets + negative_dets),
        }
        return metrics

    def loss(self, inputs, ddf):
        """
            Compute loss value.

        Args:
            inputs (dict): Dictionarry containing batched inputs. Not used here.
            ddf (torch.Tensor): DDF predicted by model.
        """

        # device = ddf.device

        inp_shape = ddf.shape
        assert (inp_shape[1] == 2 and (len(inp_shape) == 4)) or (
            inp_shape[1] == 3 and (len(inp_shape) == 5)
        ), f" Input shape {inp_shape} incorrect, expected (batch, 2, h, w) for 2D or (batch, 3, d, h, w) for 3D data"

        # Get problem dimension (2d/3d)
        dim = len(inp_shape) - 2

        if dim == 2:
            determinant = self._computeDetJac2D(ddf)
            # Border effects :
            determinant[:, 0, :] = torch.ones_like(determinant[:, 0, :])
            determinant[:, :, 0] = torch.ones_like(determinant[:, :, 0])
        else:
            determinant = self._computeDetJac3D(ddf)
            # Border effects :
            determinant[:, 0, :, :] = torch.ones_like(determinant[:, 0, :, :])
            determinant[:, :, 0, :] = torch.ones_like(determinant[:, :, 0, :])
            determinant[:, :, :, 0] = torch.ones_like(determinant[:, :, :, 0])

        # Get loss after thresholding to avoid log(<0)
        pos_det = torch.nn.Threshold(1e-9, 1e-9)(determinant)
        log_pos_det = torch.log(pos_det) ** 2
        loss = torch.mean(log_pos_det)

        return loss
