"""
(Not used in paper, for testing purposes: loss computes rigidity everywhere.)

Loss module for deformation-based learning focusing on strain rigidity.

This class computes strain rigidity loss for both 2D and 3D deformations, based on the
strain tensor's singular values after applying the deformation gradient. It can be used
for evaluating strain-related properties of the predicted displacement field (ddf).

Key Features:
    - `loss(inputs, ddf)`: computes the strain rigidity loss, using SVD of strain tensors.
    - `metric(inputs, ddf)`: evaluates strain rigidity metric for monitoring.
    - Handles both 2D and 3D cases by calculating deformation gradients and strain tensors.
    - Supports masking of specific regions to focus loss/metrics on certain areas.

Notes:
    - In 2D, the strain tensor is computed based on the displacement gradients in the x and y directions.
    - In 3D, the strain tensor additionally includes the z direction.
"""

import torch


class Rigidity:
    """
    Class to compute Strain Rigidity loss
    """

    def __init__(self, params, **kwargs):
        self.params = params

    def _computeJac2D(self, phi):
        dx = torch.gradient(phi[:, 0, :, :], axis=(1, 2))  # Warning
        dy = torch.gradient(phi[:, 1, :, :], axis=(1, 2))  # Warning

        phiX_dx, phiX_dy = dx
        phiY_dx, phiY_dy = dy

        return phiX_dx, phiX_dy, phiY_dx, phiY_dy

    def _computeStrainTensor2D(self, phiX_dx, phiX_dy, phiY_dx, phiY_dy):
        ST_x = phiX_dx
        ST_y = phiY_dy
        ST_xy = (phiX_dy + phiY_dx) / 2
        return ST_x, ST_y, ST_xy

    def _computeJac3D(self, phi):
        dx = torch.gradient(phi[:, 0, :, :, :], axis=(1, 2, 3))
        dy = torch.gradient(phi[:, 1, :, :, :], axis=(1, 2, 3))
        dz = torch.gradient(phi[:, 2, :, :, :], axis=(1, 2, 3))

        phiX_dx, phiX_dy, phiX_dz = dx
        phiY_dx, phiY_dy, phiY_dz = dy
        phiZ_dx, phiZ_dy, phiZ_dz = dz

        return (
            phiX_dx,
            phiX_dy,
            phiX_dz,
            phiY_dx,
            phiY_dy,
            phiY_dz,
            phiZ_dx,
            phiZ_dy,
            phiZ_dz,
        )

    def _computeStrainTensor3D(
        self,
        phiX_dx,
        phiX_dy,
        phiX_dz,
        phiY_dx,
        phiY_dy,
        phiY_dz,
        phiZ_dx,
        phiZ_dy,
        phiZ_dz,
    ):
        ST_x = phiX_dx
        ST_y = phiY_dy
        ST_z = phiZ_dz

        ST_xy = (phiX_dy + phiY_dx) / 2
        ST_xz = (phiX_dz + phiZ_dx) / 2
        ST_yz = (phiY_dz + phiZ_dy) / 2

        return ST_x, ST_y, ST_z, ST_xy, ST_xz, ST_yz

    def loss(self, inputs, ddf):
        """
            Compute loss value.

        Args:
            inputs (dict): Dictionarry containing batched inputs.
            ddf (torch.Tensor): DDF predicted by model.
            model (nn.Module): model.
        """

        # device = ddf.device

        inp_shape = ddf.shape
        assert (inp_shape[1] == 2 and (len(inp_shape) == 4)) or (
            inp_shape[1] == 3 and (len(inp_shape) == 5)
        ), f" Input shape {inp_shape} incorrect, expected (batch, 2, h, w) for 2D or (batch, 3, d, h, w) for 3D data"

        # Get problem dimension (2d/3d)
        dim = len(inp_shape) - 2

        if dim == 2:
            phiX_dx, phiX_dy, phiY_dx, phiY_dy = self._computeJac2D(ddf)
            # # TODO Might remove since we replace borders by zeros later? check my code DetJac loss
            phiX_dx, phiX_dy = phiX_dx[:, 1:-1, 1:-1], phiX_dy[:, 1:-1, 1:-1]
            phiY_dx, phiY_dy = phiY_dx[:, 1:-1, 1:-1], phiY_dy[:, 1:-1, 1:-1]
            ST_x, ST_y, ST_xy = self._computeStrainTensor2D(
                phiX_dx, phiX_dy, phiY_dx, phiY_dy
            )
            stacked = torch.stack(
                [
                    ST_x.reshape(-1, 1),
                    ST_xy.reshape(-1, 1),
                    ST_xy.reshape(-1, 1),
                    ST_y.reshape(-1, 1),
                ],
                1,
            ).reshape(-1, 2, 2)
        else:
            (
                phiX_dx,
                phiX_dy,
                phiX_dz,
                phiY_dx,
                phiY_dy,
                phiY_dz,
                phiZ_dx,
                phiZ_dy,
                phiZ_dz,
            ) = self._computeJac3D(ddf)
            # # TODO Might remove since we replace borders by zeros later? check my code DetJac loss
            phiX_dx, phiX_dy, phiX_dz = (
                phiX_dx[:, 1:-1, 1:-1, 1:-1],
                phiX_dy[:, 1:-1, 1:-1, 1:-1],
                phiX_dz[:, 1:-1, 1:-1, 1:-1],
            )
            phiY_dx, phiY_dy, phiY_dz = (
                phiY_dx[:, 1:-1, 1:-1, 1:-1],
                phiY_dy[:, 1:-1, 1:-1, 1:-1],
                phiY_dz[:, 1:-1, 1:-1, 1:-1],
            )
            phiZ_dx, phiZ_dy, phiZ_dz = (
                phiZ_dx[:, 1:-1, 1:-1, 1:-1],
                phiZ_dy[:, 1:-1, 1:-1, 1:-1],
                phiZ_dz[:, 1:-1, 1:-1, 1:-1],
            )
            ST_x, ST_y, ST_z, ST_xy, ST_xz, ST_yz = self._computeStrainTensor3D(
                phiX_dx,
                phiX_dy,
                phiX_dz,
                phiY_dx,
                phiY_dy,
                phiY_dz,
                phiZ_dx,
                phiZ_dy,
                phiZ_dz,
            )
            stacked = torch.stack(
                [
                    ST_x.reshape(-1, 1),
                    ST_xy.reshape(-1, 1),
                    ST_xz.reshape(-1, 1),
                    ST_xy.reshape(-1, 1),
                    ST_y.reshape(-1, 1),
                    ST_yz.reshape(-1, 1),
                    ST_xz.reshape(-1, 1),
                    ST_yz.reshape(-1, 1),
                    ST_z.reshape(-1, 1),
                ],
                1,
            ).reshape(-1, 3, 3)
        svd_ST = torch.linalg.svd(stacked)
        mean_squared_svd = torch.sum(torch.square(svd_ST.S), 1)
        loss = torch.mean(mean_squared_svd)  # Over batch and coordinates
        return loss

    def metric(self, inputs, ddf):
        device = ddf.device
        inp_shape = ddf.shape
        assert (inp_shape[1] == 2 and (len(inp_shape) == 4)) or (
            inp_shape[1] == 3 and (len(inp_shape) == 5)
        ), f" Input shape {inp_shape} incorrect, expected (batch, 2, h, w) for 2D or (batch, 3, d, h, w) for 3D data"

        # Get problem dimension (2d/3d)
        dim = len(inp_shape) - 2

        loss_mask = inputs["loss_mask"].squeeze(1).to(device)
        mask_strain = loss_mask == 1

        if dim == 2:
            phiX_dx, phiX_dy, phiY_dx, phiY_dy = self._computeJac2D(ddf)
            ST_x, ST_y, ST_xy = self._computeStrainTensor2D(
                phiX_dx[mask_strain],
                phiX_dy[mask_strain],
                phiY_dx[mask_strain],
                phiY_dy[mask_strain],
            )
            stacked = torch.stack(
                [
                    ST_x.reshape(-1, 1),
                    ST_xy.reshape(-1, 1),
                    ST_xy.reshape(-1, 1),
                    ST_y.reshape(-1, 1),
                ],
                1,
            ).reshape(-1, 2, 2)
        else:
            (
                phiX_dx,
                phiX_dy,
                phiX_dz,
                phiY_dx,
                phiY_dy,
                phiY_dz,
                phiZ_dx,
                phiZ_dy,
                phiZ_dz,
            ) = self._computeJac3D(ddf)
            ST_x, ST_y, ST_z, ST_xy, ST_xz, ST_yz = self._computeStrainTensor3D(
                phiX_dx[mask_strain],
                phiX_dy[mask_strain],
                phiX_dz[mask_strain],
                phiY_dx[mask_strain],
                phiY_dy[mask_strain],
                phiY_dz[mask_strain],
                phiZ_dx[mask_strain],
                phiZ_dy[mask_strain],
                phiZ_dz[mask_strain],
            )
            stacked = torch.stack(
                [
                    ST_x.reshape(-1, 1),
                    ST_xy.reshape(-1, 1),
                    ST_xz.reshape(-1, 1),
                    ST_xy.reshape(-1, 1),
                    ST_y.reshape(-1, 1),
                    ST_yz.reshape(-1, 1),
                    ST_xz.reshape(-1, 1),
                    ST_yz.reshape(-1, 1),
                    ST_z.reshape(-1, 1),
                ],
                1,
            ).reshape(-1, 3, 3)
        svd_ST = torch.linalg.svd(stacked)
        mean_squared_svd = torch.sum(torch.square(svd_ST.S), 1)
        loss = torch.mean(mean_squared_svd)  # Over batch and coordinates
        metrics = {"strain_in": loss.item()}
        return metrics
