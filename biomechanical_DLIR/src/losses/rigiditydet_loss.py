"""
(Loss used in the paper.)

Loss module for deformation-based learning focused on Jacobian determinant and strain.

Requires a mask indicating where to evaluate each loss term on the DDF

This class computes a composite loss for deformation fields, combining:
    - The log-squared penalty on the determinant of the Jacobian (DetJac), encouraging volume preservation and preventing foldings.
    - A strain-based loss, computed from the symmetric part of the displacement gradient (strain tensor), encouraging smooth and plausible deformations.

Key Features:
    - `loss(inputs, ddf)`: Computes the loss using a mask to separate DetJac and strain regions.
    - `metric(inputs, ddf)`: Placeholder for future implementation of relevant metrics.
    - Handles both 2D and 3D deformations.
    - Internally computes deformation gradients and strain tensors via finite differences.
    - Supports masking to apply different losses on different spatial regions.

Notes:
    - The deformation field `ddf` is assumed to be added to the identity map (i.e., u = φ + id).
    - Gradients are computed using `torch.gradient`, with assumptions about tensor layout: (B, 2, H, W) for 2D and (B, 3, D, H, W) for 3D.
    - The strain tensor is symmetrized and its singular values are penalized to discourage local distortions.
"""

import torch


class RigidityDet:
    """
    Class to compute Detemrinant of Jacobian loss
    """

    def __init__(self, params, **kwargs):
        self.params = params

    def _computeJac2D(self, phi):
        dx = torch.gradient(phi[:, 0, :, :], axis=(1, 2))  # Warning
        dy = torch.gradient(phi[:, 1, :, :], axis=(1, 2))  # Warning

        phiX_dx, phiX_dy = dx
        phiY_dx, phiY_dy = dy

        # # # TODO Might remove since we replace borders by zeros later? check my code DetJac loss
        # phiX_dx, phiX_dy = phiX_dx[:,1:-1,1:-1], phiX_dy[:,1:-1,1:-1]
        # phiY_dx, phiY_dy = phiY_dx[:,1:-1,1:-1], phiY_dy[:,1:-1,1:-1]

        # # TODO +1 because u = phi + id ?
        # phiX_dx +=1
        # phiY_dy +=1

        return phiX_dx, phiX_dy, phiY_dx, phiY_dy

    def _computeDetJac2D(self, phiX_dx, phiX_dy, phiY_dx, phiY_dy):
        # TODO +1 because u = phi + id
        phiX_dx += 1
        phiY_dy += 1
        determinant = phiX_dx * phiY_dy - phiY_dx * phiX_dy
        return determinant

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

        # phiX_dx, phiX_dy, phiX_dz = phiX_dx[:,1:-1,1:-1,1:-1], phiX_dy[:,1:-1,1:-1,1:-1], phiX_dz[:,1:-1,1:-1,1:-1]
        # phiY_dx, phiY_dy, phiY_dz = phiY_dx[:,1:-1,1:-1,1:-1], phiY_dy[:,1:-1,1:-1,1:-1], phiY_dz[:,1:-1,1:-1,1:-1]
        # phiZ_dx, phiZ_dy, phiZ_dz = phiZ_dx[:,1:-1,1:-1,1:-1], phiZ_dy[:,1:-1,1:-1,1:-1], phiZ_dz[:,1:-1,1:-1,1:-1]

        # # TODO +1 because u = phi + id
        # phiX_dx +=1
        # phiY_dy +=1
        # phiZ_dz +=1
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

    def _computeDetJac3D(
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
        # TODO +1 because u = phi + id
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

    def metric(self, inputs, ddf):
        # TODO
        return {}

    def loss(self, inputs, ddf):
        """
            Compute loss value.

        Args:
            inputs (dict): Dictionarry containing batched inputs.
            ddf (torch.Tensor): DDF predicted by model.
            model (nn.Module): model.
        """

        device = ddf.device

        inp_shape = ddf.shape
        assert (inp_shape[1] == 2 and (len(inp_shape) == 4)) or (
            inp_shape[1] == 3 and (len(inp_shape) == 5)
        ), f" Input shape {inp_shape} incorrect, expected (batch, 2, h, w) for 2D or (batch, 3, d, h, w) for 3D data"

        # Get problem dimension (2d/3d)
        dim = len(inp_shape) - 2

        # 0 means DetJac loss / 1 means rigidity loss
        loss_mask = inputs["loss_mask"].squeeze(1).to(device)
        mask_det = loss_mask == 0
        mask_strain = loss_mask == 1

        if dim == 2:
            phiX_dx, phiX_dy, phiY_dx, phiY_dy = self._computeJac2D(ddf)

            # Det Loss :
            determinant = self._computeDetJac2D(
                phiX_dx[mask_det],
                phiX_dy[mask_det],
                phiY_dx[mask_det],
                phiY_dy[mask_det],
            )
            # Strain loss:
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
            # Det Loss :
            determinant = self._computeDetJac3D(
                phiX_dx[mask_det],
                phiX_dy[mask_det],
                phiX_dz[mask_det],
                phiY_dx[mask_det],
                phiY_dy[mask_det],
                phiY_dz[mask_det],
                phiZ_dx[mask_det],
                phiZ_dy[mask_det],
                phiZ_dz[mask_det],
            )
            # Strain loss:
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

        loss = 0.0
        if mask_det.max() > 0:
            pos_det = torch.nn.Threshold(1e-9, 1e-9)(determinant)
            log_pos_det = torch.log(pos_det) ** 2
            loss += torch.mean(log_pos_det)
        if mask_strain.max() > 0:
            # Strain loss:
            svd_ST = torch.linalg.svd(stacked)
            mean_squared_svd = torch.sum(torch.square(svd_ST.S), 1)
            loss += torch.mean(mean_squared_svd)  # Over batch and coordinates
        return loss
