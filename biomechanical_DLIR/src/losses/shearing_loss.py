import torch

class Shearing:
    """
    Class to compute Shearing loss
    """
    def __init__(self,params, **kwargs):
        self.params = params
        self.weight_residual_disp = 0.0 # TODO
        
    def _computeJac2D(self, phi):
        dx = torch.gradient(phi[:,0,:,:], axis=(1,2)) # Warning
        dy = torch.gradient(phi[:,1,:,:], axis=(1,2)) # Warning 
          
        phiX_dx, phiX_dy = dx
        phiY_dx, phiY_dy = dy

        return phiX_dx, phiX_dy, phiY_dx, phiY_dy
    def _computeStrainTensor2D(self, phiX_dx, phiX_dy, phiY_dx, phiY_dy):
        ST_x = phiX_dx
        ST_y = phiY_dy
        ST_xy = (phiX_dy + phiY_dx)/2
        return ST_x, ST_y, ST_xy
    def _projectDisplacement2D_vector(self,phi,projection_vector):
        # Normalize projection vector:
        normalized_vect = projection_vector/torch.linalg.norm(projection_vector,dim=-1).unsqueeze(1)
        # Compute Scalar product 
        projected_amplitudes = torch.einsum('bdHW, bd -> bHW', phi, normalized_vect)
        # Compute projection
        projected_displacements = torch.einsum('bHW, bd -> bdHW', projected_amplitudes,normalized_vect)
        return projected_displacements
    def _projectDisplacement2D_image(self,phi,projection_vector):
        """
            Projects phi on projection vector, where projection vector is of same shape as phi.
        """
        # Normalize projection vector:
        normalized_vect = projection_vector/torch.linalg.norm(projection_vector,dim=1).unsqueeze(1)
        # Using image as projection vector will introduce NaN's at voxels where no projection vector is given...
        normalized_vect = torch.nan_to_num(normalized_vect, nan=0.0) 
        # Compute Scalar product 
        projected_amplitudes = torch.einsum('bdHW, bdHW -> bHW', phi, normalized_vect)
        projected_displacements= projected_amplitudes.unsqueeze(1)*normalized_vect
        return projected_displacements
    def _projectDisplacement2D(self,phi,projection_vector):
        if projection_vector.shape == phi.shape:
            projected_displacements = self._projectDisplacement2D_image(phi,projection_vector)
        else:
            projected_displacements = self._projectDisplacement2D_vector(phi,projection_vector)
        return projected_displacements    

    def _computeJac3D(self, phi):
        dx = torch.gradient(phi[:,0,:,:,:], axis=(1,2,3))
        dy = torch.gradient(phi[:,1,:,:,:], axis=(1,2,3))
        dz = torch.gradient(phi[:,2,:,:,:], axis=(1,2,3))
        
        phiX_dx, phiX_dy, phiX_dz = dx
        phiY_dx, phiY_dy, phiY_dz = dy
        phiZ_dx, phiZ_dy, phiZ_dz = dz

        return phiX_dx, phiX_dy, phiX_dz, phiY_dx, phiY_dy, phiY_dz, phiZ_dx, phiZ_dy, phiZ_dz  
    def _computeStrainTensor3D(self,phiX_dx, phiX_dy, phiX_dz, phiY_dx, phiY_dy, phiY_dz, phiZ_dx, phiZ_dy, phiZ_dz):
        ST_x = phiX_dx
        ST_y = phiY_dy
        ST_z = phiZ_dz

        ST_xy = (phiX_dy + phiY_dx)/2
        ST_xz = (phiX_dz + phiZ_dx)/2
        ST_yz = (phiY_dz + phiZ_dy)/2

        return ST_x,ST_y,ST_z, ST_xy, ST_xz, ST_yz  
    
    def _projectDisplacement3D_vector(self,phi,projection_vector):
        # Normalize projection vector:
        normalized_vect = projection_vector/torch.linalg.norm(projection_vector,dim=-1).unsqueeze(1)
        # Compute Scalar product 
        projected_amplitudes = torch.einsum('bdHWD, bd -> bHWD', phi, normalized_vect)
        # Compute projection
        projected_displacements = torch.einsum('bHWD, bd -> bdHWD', projected_amplitudes,normalized_vect)
        return projected_displacements
    
    def _projectDisplacement3D_image(self,phi,projection_vector):
        """
            Projects phi on projection vector, where projection vector is of same shape as phi.
        """
        # Normalize projection vector:
        normalized_vect = projection_vector/torch.linalg.norm(projection_vector,dim=1).unsqueeze(1)
        # Using image as projection vector will introduce NaN's at voxels where no projection vector is given...
        normalized_vect = torch.nan_to_num(normalized_vect, nan=0.0) 
        # Compute Scalar product 
        projected_amplitudes = torch.einsum('bdHWD, bdHWD -> bHWD', phi, normalized_vect)
        projected_displacements= projected_amplitudes.unsqueeze(1)*normalized_vect
        return projected_displacements
    
    def _projectDisplacement3D(self,phi,projection_vector):
        if projection_vector.shape == phi.shape:
            projected_displacements = self._projectDisplacement3D_image(phi,projection_vector)
        else:
            projected_displacements = self._projectDisplacement3D_vector(phi,projection_vector)
        return projected_displacements


    def loss(self,inputs, ddf):
        """
            Compute loss value.

        Args:
            inputs (dict): Dictionarry containing batched inputs.
            ddf (torch.Tensor): DDF predicted by model.
            model (nn.Module): model.
        """

        device = ddf.device
        projection_vector = inputs["proj_vector"].to(device)

        inp_shape = ddf.shape
        assert (inp_shape[1]==2 and (len(inp_shape) == 4) )  or (inp_shape[1]==3 and (len(inp_shape) == 5) ), f" Input shape {inp_shape} incorrect, expected (batch, 2, h, w) for 2D or (batch, 3, d, h, w) for 3D data"
        
        # Get problem dimension (2d/3d)
        dim = len(inp_shape) - 2

        if dim == 2 :
            projected_displacements = self._projectDisplacement2D(ddf, projection_vector)
            residual_displacements = ddf - projected_displacements
            # On Projected : 
            phiX_dx_proj, phiX_dy_proj, phiY_dx_proj, phiY_dy_proj = self._computeJac2D(projected_displacements)
            # # TODO Might remove since we replace borders by zeros later? check my code DetJac loss 
            phiX_dx_proj, phiX_dy_proj = phiX_dx_proj[:,1:-1,1:-1], phiX_dy_proj[:,1:-1,1:-1]
            phiY_dx_proj, phiY_dy_proj = phiY_dx_proj[:,1:-1,1:-1], phiY_dy_proj[:,1:-1,1:-1]

            ST_x_proj,ST_y_proj,ST_xy_proj = self._computeStrainTensor2D(phiX_dx_proj, phiX_dy_proj,
                                                                        phiY_dx_proj, phiY_dy_proj)
            stacked_proj = torch.stack([ST_x_proj.reshape(-1,1),ST_xy_proj.reshape(-1,1),
                                   ST_xy_proj.reshape(-1,1),ST_y_proj.reshape(-1,1)],1).reshape(-1,2,2)
            
            # On Residual : 
            phiX_dx_res, phiX_dy_res, phiY_dx_res, phiY_dy_res = self._computeJac2D(residual_displacements)
            # # TODO Might remove since we replace borders by zeros later? check my code DetJac loss 
            phiX_dx_res, phiX_dy_res = phiX_dx_res[:,1:-1,1:-1], phiX_dy_res[:,1:-1,1:-1]
            phiY_dx_res, phiY_dy_res = phiY_dx_res[:,1:-1,1:-1], phiY_dy_res[:,1:-1,1:-1]

            ST_x_res,ST_y_res,ST_xy_res = self._computeStrainTensor2D(phiX_dx_res, phiX_dy_res,
                                                                    phiY_dx_res, phiY_dy_res)
            stacked_res = torch.stack([ST_x_res.reshape(-1,1),ST_xy_res.reshape(-1,1),
                                   ST_xy_res.reshape(-1,1),ST_y_res.reshape(-1,1)],1).reshape(-1,2,2)
        else:
            projected_displacements = self._projectDisplacement3D(ddf, projection_vector)
            residual_displacements = ddf - projected_displacements
            # On Projected : 
            phiX_dx_proj, phiX_dy_proj, phiX_dz_proj, phiY_dx_proj, phiY_dy_proj, phiY_dz_proj, phiZ_dx_proj, phiZ_dy_proj, phiZ_dz_proj = self._computeJac3D(projected_displacements)
            # # TODO Might remove since we replace borders by zeros later? check my code DetJac loss 
            phiX_dx_proj, phiX_dy_proj, phiX_dz_proj = phiX_dx_proj[:,1:-1,1:-1,1:-1], phiX_dy_proj[:,1:-1,1:-1,1:-1], phiX_dz_proj[:,1:-1,1:-1,1:-1]
            phiY_dx_proj, phiY_dy_proj, phiY_dz_proj = phiY_dx_proj[:,1:-1,1:-1,1:-1], phiY_dy_proj[:,1:-1,1:-1,1:-1], phiY_dz_proj[:,1:-1,1:-1,1:-1]
            phiZ_dx_proj, phiZ_dy_proj, phiZ_dz_proj = phiZ_dx_proj[:,1:-1,1:-1,1:-1], phiZ_dy_proj[:,1:-1,1:-1,1:-1], phiZ_dz_proj[:,1:-1,1:-1,1:-1]
            ST_x_proj,ST_y_proj,ST_z_proj, ST_xy_proj, ST_xz_proj, ST_yz_proj = self._computeStrainTensor3D(phiX_dx_proj, phiX_dy_proj, phiX_dz_proj,
                                                                            phiY_dx_proj, phiY_dy_proj, phiY_dz_proj,
                                                                            phiZ_dx_proj, phiZ_dy_proj, phiZ_dz_proj)
            stacked_proj = torch.stack([ST_x_proj.reshape(-1,1),ST_xy_proj.reshape(-1,1),ST_xz_proj.reshape(-1,1),
                       ST_xy_proj.reshape(-1,1),ST_y_proj.reshape(-1,1),ST_yz_proj.reshape(-1,1),
                       ST_xz_proj.reshape(-1,1),ST_yz_proj.reshape(-1,1),ST_z_proj.reshape(-1,1)],1).reshape(-1,3,3)

            # On Residual : 
            phiX_dx_res, phiX_dy_res, phiX_dz_res, phiY_dx_res, phiY_dy_res, phiY_dz_res, phiZ_dx_res, phiZ_dy_res, phiZ_dz_res = self._computeJac3D(residual_displacements)
            # # TODO Might remove since we replace borders by zeros later? check my code DetJac loss 
            phiX_dx_res, phiX_dy_res, phiX_dz_res = phiX_dx_res[:,1:-1,1:-1,1:-1], phiX_dy_res[:,1:-1,1:-1,1:-1], phiX_dz_res[:,1:-1,1:-1,1:-1]
            phiY_dx_res, phiY_dy_res, phiY_dz_res = phiY_dx_res[:,1:-1,1:-1,1:-1], phiY_dy_res[:,1:-1,1:-1,1:-1], phiY_dz_res[:,1:-1,1:-1,1:-1]
            phiZ_dx_res, phiZ_dy_res, phiZ_dz_res = phiZ_dx_res[:,1:-1,1:-1,1:-1], phiZ_dy_res[:,1:-1,1:-1,1:-1], phiZ_dz_res[:,1:-1,1:-1,1:-1]
            ST_x_res,ST_y_res,ST_z_res, ST_xy_res, ST_xz_res, ST_yz_res = self._computeStrainTensor3D(phiX_dx_res, phiX_dy_res, phiX_dz_res,
                                                                            phiY_dx_res, phiY_dy_res, phiY_dz_res,
                                                                            phiZ_dx_res, phiZ_dy_res, phiZ_dz_res)
            stacked_res = torch.stack([ST_x_res.reshape(-1,1),ST_xy_res.reshape(-1,1),ST_xz_res.reshape(-1,1),
                       ST_xy_res.reshape(-1,1),ST_y_res.reshape(-1,1),ST_yz_res.reshape(-1,1),
                       ST_xz_res.reshape(-1,1),ST_yz_res.reshape(-1,1),ST_z_res.reshape(-1,1)],1).reshape(-1,3,3)
            
        svd_ST_proj=torch.linalg.svd(stacked_proj)
        svd_ST_res=torch.linalg.svd(stacked_res)
        mean_squared_svd = torch.sum(torch.square(svd_ST_proj.S),1) + self.weight_residual_disp*torch.mean(torch.square(svd_ST_res.S) ,1)
        loss = torch.mean(mean_squared_svd)
        return loss