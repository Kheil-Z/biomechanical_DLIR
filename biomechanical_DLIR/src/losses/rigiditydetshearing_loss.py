"""

(Loss used in the paper.)

Loss module for deformation-based learning focusing on strain,, shearing and determinant Jacobian.

Requires a mask indicating where to evaluate each loss term on the DDF

This class computes the loss related to the determinant of the Jacobian, strain, and shearing 
for both 2D and 3D deformations. It is designed to evaluate how the predicted displacement field 
affects the volume change, strain, and shear displacement within the deformation.

Key Features:
    - `loss(inputs, ddf)`: Computes the total loss, combining determinant of Jacobian, strain, 
      and shearing components.
    - `metric(inputs, ddf)`: Evaluates the relevant metric values for the deformation field.
    - Supports both 2D and 3D cases, handling deformation gradients and strain/shearing tensor calculations.
    - Supports masking to apply different losses on different spatial regions.
    - Can be customized with a weight for the residual displacement in the shear loss component.

Notes:
    - In 2D, the strain and shear loss terms are computed from the deformation gradients in the x and y directions.
    - In 3D, the strain and shear loss terms incorporate the additional z direction.
"""

import torch

class RigidityDetShearing:
    """
    Class to compute Detemrinant of Jacobian loss
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
    
    def _computeDetJac2D(self, phiX_dx, phiX_dy, phiY_dx, phiY_dy):
        # TODO +1 because u = phi + id
        phiX_dx +=1
        phiY_dy +=1
        determinant = phiX_dx*phiY_dy - phiY_dx*phiX_dy
        return determinant
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
    
    def _computeDetJac3D(self, phiX_dx, phiX_dy, phiX_dz, phiY_dx, phiY_dy, phiY_dz, phiZ_dx, phiZ_dy, phiZ_dz):
        phiX_dx +=1
        phiY_dy +=1
        phiZ_dz +=1
        plus = (phiX_dx * phiY_dy * phiZ_dz) + (phiX_dy * phiY_dz * phiZ_dx) + (phiX_dz * phiY_dx * phiZ_dy)  
        minus = (phiX_dz * phiY_dy * phiZ_dx) + (phiX_dy * phiY_dx * phiZ_dz) + (phiX_dx * phiY_dz * phiZ_dy)  
        determinant = plus - minus
            #         determinant[:,0,:] = torch.ones_like(determinant[:,0,:])
            # determinant[:,:,0] = torch.ones_like(determinant[:,:,0])
        return determinant
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
        
        inp_shape = ddf.shape
        assert (inp_shape[1]==2 and (len(inp_shape) == 4) )  or (inp_shape[1]==3 and (len(inp_shape) == 5) ), f" Input shape {inp_shape} incorrect, expected (batch, 2, h, w) for 2D or (batch, 3, d, h, w) for 3D data"
        
        # Get problem dimension (2d/3d)
        dim = len(inp_shape) - 2

        # 0 means DetJac loss / 1 means rigidity loss / 2 means shearing loss
        loss_mask = inputs["loss_mask"].squeeze(1).to(device)
        mask_det = loss_mask==0
        mask_strain = loss_mask==1
        mask_shear = loss_mask==2

        projection_vector = inputs["proj_vector"].to(device)

        if dim == 2 :
            phiX_dx, phiX_dy, phiY_dx, phiY_dy = self._computeJac2D(ddf)

            ####### 1) Det Loss :
            determinant = self._computeDetJac2D(phiX_dx[mask_det], phiX_dy[mask_det],
                                                phiY_dx[mask_det], phiY_dy[mask_det])
            ####### 2) Strain loss:
            ST_x,ST_y,ST_xy = self._computeStrainTensor2D(phiX_dx[mask_strain], phiX_dy[mask_strain],
                                                        phiY_dx[mask_strain], phiY_dy[mask_strain])
            stacked = torch.stack([ST_x.reshape(-1,1),ST_xy.reshape(-1,1),
                                   ST_xy.reshape(-1,1),ST_y.reshape(-1,1)],1).reshape(-1,2,2)
            
            ####### 3) Shear loss: 
            projected_displacements = self._projectDisplacement2D(ddf, projection_vector)
            residual_displacements = ddf - projected_displacements
            # On Projected : 
            phiX_dx_proj, phiX_dy_proj, phiY_dx_proj, phiY_dy_proj = self._computeJac2D(projected_displacements)
            ST_x_proj,ST_y_proj,ST_xy_proj = self._computeStrainTensor2D(phiX_dx_proj[mask_shear], phiX_dy_proj[mask_shear],
                                                                        phiY_dx_proj[mask_shear], phiY_dy_proj[mask_shear])
            stacked_proj = torch.stack([ST_x_proj.reshape(-1,1),ST_xy_proj.reshape(-1,1),
                                   ST_xy_proj.reshape(-1,1),ST_y_proj.reshape(-1,1)],1).reshape(-1,2,2)
            
            # On Residual : 
            phiX_dx_res, phiX_dy_res, phiY_dx_res, phiY_dy_res = self._computeJac2D(residual_displacements)
            ST_x_res,ST_y_res,ST_xy_res = self._computeStrainTensor2D(phiX_dx_res[mask_shear], phiX_dy_res[mask_shear],
                                                                    phiY_dx_res[mask_shear], phiY_dy_res[mask_shear])
            stacked_res = torch.stack([ST_x_res.reshape(-1,1),ST_xy_res.reshape(-1,1),
                                   ST_xy_res.reshape(-1,1),ST_y_res.reshape(-1,1)],1).reshape(-1,2,2)
        else:
            phiX_dx, phiX_dy, phiX_dz, phiY_dx, phiY_dy, phiY_dz, phiZ_dx, phiZ_dy, phiZ_dz = self._computeJac3D(ddf)
            ####### 1) Det Loss :
            determinant = self._computeDetJac3D(phiX_dx[mask_det], phiX_dy[mask_det], phiX_dz[mask_det],
                                                phiY_dx[mask_det], phiY_dy[mask_det], phiY_dz[mask_det],
                                                phiZ_dx[mask_det], phiZ_dy[mask_det], phiZ_dz[mask_det])
            ####### 2) Strain loss:
            ST_x,ST_y,ST_z, ST_xy, ST_xz, ST_yz = self._computeStrainTensor3D(phiX_dx[mask_strain], phiX_dy[mask_strain], phiX_dz[mask_strain],
                                                                            phiY_dx[mask_strain], phiY_dy[mask_strain], phiY_dz[mask_strain],
                                                                            phiZ_dx[mask_strain], phiZ_dy[mask_strain], phiZ_dz[mask_strain])
            stacked = torch.stack([ST_x.reshape(-1,1),ST_xy.reshape(-1,1),ST_xz.reshape(-1,1),
                                   ST_xy.reshape(-1,1),ST_y.reshape(-1,1),ST_yz.reshape(-1,1),
                                   ST_xz.reshape(-1,1),ST_yz.reshape(-1,1),ST_z.reshape(-1,1)],1).reshape(-1,3,3)
            
            ####### 3) Shear loss: 
            projected_displacements = self._projectDisplacement3D(ddf, projection_vector)
            residual_displacements = ddf - projected_displacements
            # On Projected : 
            phiX_dx_proj, phiX_dy_proj, phiX_dz_proj, phiY_dx_proj, phiY_dy_proj, phiY_dz_proj, phiZ_dx_proj, phiZ_dy_proj, phiZ_dz_proj = self._computeJac3D(projected_displacements)
            ST_x_proj,ST_y_proj,ST_z_proj, ST_xy_proj, ST_xz_proj, ST_yz_proj = self._computeStrainTensor3D(phiX_dx_proj[mask_shear], phiX_dy_proj[mask_shear], phiX_dz_proj[mask_shear],
                                                                            phiY_dx_proj[mask_shear], phiY_dy_proj[mask_shear], phiY_dz_proj[mask_shear],
                                                                            phiZ_dx_proj[mask_shear], phiZ_dy_proj[mask_shear], phiZ_dz_proj[mask_shear])
            stacked_proj = torch.stack([ST_x_proj.reshape(-1,1),ST_xy_proj.reshape(-1,1),ST_xz_proj.reshape(-1,1),
                       ST_xy_proj.reshape(-1,1),ST_y_proj.reshape(-1,1),ST_yz_proj.reshape(-1,1),
                       ST_xz_proj.reshape(-1,1),ST_yz_proj.reshape(-1,1),ST_z_proj.reshape(-1,1)],1).reshape(-1,3,3)

            # On Residual : 
            phiX_dx_res, phiX_dy_res, phiX_dz_res, phiY_dx_res, phiY_dy_res, phiY_dz_res, phiZ_dx_res, phiZ_dy_res, phiZ_dz_res = self._computeJac3D(residual_displacements)
            ST_x_res,ST_y_res,ST_z_res, ST_xy_res, ST_xz_res, ST_yz_res = self._computeStrainTensor3D(phiX_dx_res[mask_shear], phiX_dy_res[mask_shear], phiX_dz_res[mask_shear],
                                                                            phiY_dx_res[mask_shear], phiY_dy_res[mask_shear], phiY_dz_res[mask_shear],
                                                                            phiZ_dx_res[mask_shear], phiZ_dy_res[mask_shear], phiZ_dz_res[mask_shear])
            stacked_res = torch.stack([ST_x_res.reshape(-1,1),ST_xy_res.reshape(-1,1),ST_xz_res.reshape(-1,1),
                       ST_xy_res.reshape(-1,1),ST_y_res.reshape(-1,1),ST_yz_res.reshape(-1,1),
                       ST_xz_res.reshape(-1,1),ST_yz_res.reshape(-1,1),ST_z_res.reshape(-1,1)],1).reshape(-1,3,3)
        
        loss = 0.
        # 1) Det
        if mask_det.max()>0:
            pos_det = torch.nn.Threshold(1e-9, 1e-9)(determinant)
            log_pos_det = torch.log(pos_det)**2
            loss += torch.mean(log_pos_det)

        # 2) Strain loss:
        if mask_strain.max()>0:
            svd_ST=torch.linalg.svd(stacked)
            mean_squared_svd = torch.sum(torch.square(svd_ST.S),1)
            loss += torch.mean(mean_squared_svd) # Over batch and coordinates

        # 3) Shearing loss:
        if mask_shear.max()>0:
            svd_ST_proj=torch.linalg.svd(stacked_proj)
            svd_ST_res=torch.linalg.svd(stacked_res)
            mean_squared_svd_shearing = torch.sum(torch.square(svd_ST_proj.S),1) + self.weight_residual_disp*torch.mean(torch.square(svd_ST_res.S) ,1)
            loss += torch.mean(mean_squared_svd_shearing) # Over batch and coordinates
        return loss
    

    def metric(self,inputs, ddf):
        device = ddf.device
        inp_shape = ddf.shape
        assert (inp_shape[1]==2 and (len(inp_shape) == 4) )  or (inp_shape[1]==3 and (len(inp_shape) == 5) ), f" Input shape {inp_shape} incorrect, expected (batch, 2, h, w) for 2D or (batch, 3, d, h, w) for 3D data"
        
        # Get problem dimension (2d/3d)
        dim = len(inp_shape) - 2
        
        loss_mask = inputs["loss_mask"].squeeze(1).to(device)
        mask_strain = loss_mask==1
        mask_shear = loss_mask==2
        mask_foldings = loss_mask<2

        projection_vector = inputs["proj_vector"].to(device)

        if dim == 2 :
            phiX_dx, phiX_dy, phiY_dx, phiY_dy = self._computeJac2D(ddf)
            ####### 1) Det Loss :
            determinant = self._computeDetJac2D(phiX_dx[mask_foldings], phiX_dy[mask_foldings],
                                                phiY_dx[mask_foldings], phiY_dy[mask_foldings])
            
            ####### 1) Strain (Rigidity) loss: 
            ST_x,ST_y,ST_xy = self._computeStrainTensor2D(phiX_dx[mask_strain], phiX_dy[mask_strain],
                                                        phiY_dx[mask_strain], phiY_dy[mask_strain])
            stacked = torch.stack([ST_x.reshape(-1,1),ST_xy.reshape(-1,1),
                                   ST_xy.reshape(-1,1),ST_y.reshape(-1,1)],1).reshape(-1,2,2)
            
            ####### 2) Shear loss: 
            projected_displacements = self._projectDisplacement2D(ddf, projection_vector)
            residual_displacements = ddf - projected_displacements
            # On Projected : 
            phiX_dx_proj, phiX_dy_proj, phiY_dx_proj, phiY_dy_proj = self._computeJac2D(projected_displacements)
            ST_x_proj,ST_y_proj,ST_xy_proj = self._computeStrainTensor2D(phiX_dx_proj[mask_shear], phiX_dy_proj[mask_shear],
                                                                        phiY_dx_proj[mask_shear], phiY_dy_proj[mask_shear])
            stacked_proj = torch.stack([ST_x_proj.reshape(-1,1),ST_xy_proj.reshape(-1,1),
                                   ST_xy_proj.reshape(-1,1),ST_y_proj.reshape(-1,1)],1).reshape(-1,2,2)
            
            # On Residual : 
            phiX_dx_res, phiX_dy_res, phiY_dx_res, phiY_dy_res = self._computeJac2D(residual_displacements)
            ST_x_res,ST_y_res,ST_xy_res = self._computeStrainTensor2D(phiX_dx_res[mask_shear], phiX_dy_res[mask_shear],
                                                                    phiY_dx_res[mask_shear], phiY_dy_res[mask_shear])
            stacked_res = torch.stack([ST_x_res.reshape(-1,1),ST_xy_res.reshape(-1,1),
                                   ST_xy_res.reshape(-1,1),ST_y_res.reshape(-1,1)],1).reshape(-1,2,2)


        else:
            phiX_dx, phiX_dy, phiX_dz, phiY_dx, phiY_dy, phiY_dz, phiZ_dx, phiZ_dy, phiZ_dz = self._computeJac3D(ddf)

            ####### 1) Det Loss :
            determinant = self._computeDetJac3D(phiX_dx[mask_foldings], phiX_dy[mask_foldings], phiX_dz[mask_foldings],
                                                phiY_dx[mask_foldings], phiY_dy[mask_foldings], phiY_dz[mask_foldings],
                                                phiZ_dx[mask_foldings], phiZ_dy[mask_foldings], phiZ_dz[mask_foldings])
            
            ####### 2) Strain (Rigidity) loss: 
            ST_x,ST_y,ST_z, ST_xy, ST_xz, ST_yz = self._computeStrainTensor3D(phiX_dx[mask_strain], phiX_dy[mask_strain], phiX_dz[mask_strain],
                                                                            phiY_dx[mask_strain], phiY_dy[mask_strain], phiY_dz[mask_strain],
                                                                            phiZ_dx[mask_strain], phiZ_dy[mask_strain], phiZ_dz[mask_strain])
            stacked = torch.stack([ST_x.reshape(-1,1),ST_xy.reshape(-1,1),ST_xz.reshape(-1,1),
                                   ST_xy.reshape(-1,1),ST_y.reshape(-1,1),ST_yz.reshape(-1,1),
                                   ST_xz.reshape(-1,1),ST_yz.reshape(-1,1),ST_z.reshape(-1,1)],1).reshape(-1,3,3)
            
            ####### 3) Shear loss: 
            projected_displacements = self._projectDisplacement3D(ddf, projection_vector)
            residual_displacements = ddf - projected_displacements
            # On Projected : 
            phiX_dx_proj, phiX_dy_proj, phiX_dz_proj, phiY_dx_proj, phiY_dy_proj, phiY_dz_proj, phiZ_dx_proj, phiZ_dy_proj, phiZ_dz_proj = self._computeJac3D(projected_displacements)
            ST_x_proj,ST_y_proj,ST_z_proj, ST_xy_proj, ST_xz_proj, ST_yz_proj = self._computeStrainTensor3D(phiX_dx_proj[mask_shear], phiX_dy_proj[mask_shear], phiX_dz_proj[mask_shear],
                                                                            phiY_dx_proj[mask_shear], phiY_dy_proj[mask_shear], phiY_dz_proj[mask_shear],
                                                                            phiZ_dx_proj[mask_shear], phiZ_dy_proj[mask_shear], phiZ_dz_proj[mask_shear])
            stacked_proj = torch.stack([ST_x_proj.reshape(-1,1),ST_xy_proj.reshape(-1,1),ST_xz_proj.reshape(-1,1),
                       ST_xy_proj.reshape(-1,1),ST_y_proj.reshape(-1,1),ST_yz_proj.reshape(-1,1),
                       ST_xz_proj.reshape(-1,1),ST_yz_proj.reshape(-1,1),ST_z_proj.reshape(-1,1)],1).reshape(-1,3,3)

            # On Residual : 
            phiX_dx_res, phiX_dy_res, phiX_dz_res, phiY_dx_res, phiY_dy_res, phiY_dz_res, phiZ_dx_res, phiZ_dy_res, phiZ_dz_res = self._computeJac3D(residual_displacements)
            ST_x_res,ST_y_res,ST_z_res, ST_xy_res, ST_xz_res, ST_yz_res = self._computeStrainTensor3D(phiX_dx_res[mask_shear], phiX_dy_res[mask_shear], phiX_dz_res[mask_shear],
                                                                            phiY_dx_res[mask_shear], phiY_dy_res[mask_shear], phiY_dz_res[mask_shear],
                                                                            phiZ_dx_res[mask_shear], phiZ_dy_res[mask_shear], phiZ_dz_res[mask_shear])
            stacked_res = torch.stack([ST_x_res.reshape(-1,1),ST_xy_res.reshape(-1,1),ST_xz_res.reshape(-1,1),
                       ST_xy_res.reshape(-1,1),ST_y_res.reshape(-1,1),ST_yz_res.reshape(-1,1),
                       ST_xz_res.reshape(-1,1),ST_yz_res.reshape(-1,1),ST_z_res.reshape(-1,1)],1).reshape(-1,3,3)
        
        # 0) Foldings no in shearing region:
        negative_dets = torch.sum(determinant < 0.0).item()
        positive_dets = torch.sum(determinant >= 0.0).item()
        std_log_jac = determinant[determinant>0].log().std().item()
        # metrics = {"std_log_jac":std_log_jac, 
        #            "negative_dets":negative_dets,
        #             "positive_dets":positive_dets,
        #             "percent_foldings":100*negative_dets/(positive_dets+negative_dets)}

        # 1) Strain (Rigidity) metric
        svd_ST=torch.linalg.svd(stacked)
        mean_squared_svd = torch.sum(torch.square(svd_ST.S),1)
        strain = torch.mean(mean_squared_svd) # Over batch and coordinates


        # 2) Shearing loss:
        svd_ST_proj=torch.linalg.svd(stacked_proj)
        mean_squared_svd_shearing_proj = torch.sum(torch.square(svd_ST_proj.S),1) 
        shearing_proj = torch.mean(mean_squared_svd_shearing_proj) # Over batch and coordinates

        svd_ST_res=torch.linalg.svd(stacked_res)
        mean_squared_svd_shearing_res = torch.mean(torch.square(svd_ST_res.S) ,1)
        shearing_res = torch.mean(mean_squared_svd_shearing_res) # Over batch and coordinates

        metrics = {
                    "std_log_jac_notStrain":std_log_jac, 
                   "negative_dets_notStrain":negative_dets,
                    "positive_dets_notStrain":positive_dets,
                    "percent_foldings_notStrain":100*negative_dets/(positive_dets+negative_dets),

                    "strain_in":strain.item(),
                   "shearing_projected":shearing_proj.item(),
                   "shearing_residual":shearing_res.item()}
        return metrics
