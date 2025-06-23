"""
Script to compute strain and shear masks on AbdomenCTCT dataset
Also returns normal vectors for specific anatomical structures to project shearing for the loss as described in paper.

The idea is to identify mechanical interactions between various body parts,
such as bones, organs, and tissues, based on anatomical labels.
It uses binary dilation to identify adjacent structures and computes local surface normals using
Principal Component Analysis (PCA) to capture the interface directions. These results are saved as NIfTI files.

Key operations performed:
1. Load and process 3D medical image data and segmentation labels.
2. Compute strain masks based on anatomical label interactions, including both rigid and shear stress movements.
3. For regions of interest (e.g., lung and vertebrae interactions), compute interface normals to characterize the mechanical strain directions.
4. Generate and save output files containing strain masks and normal projections in NIfTI format.

Dependencies:
    - Numpy
    - PyTorch
    - MONAI (for data transformation)
    - Scipy
    - Nibabel
    - scikit-learn (for PCA)
    - glob (for file searching)
    - os (for directory management)

Command-line arguments:
    --path: Path to reoriented Learn2Reg data.

Functions:
    - compute_interface_normals: Computes the surface normals for regions of strain based on local neighborhood information.
    - strain_mask: Generates a binary mask indicating where two labels are in close proximity (representing mechanical strain).

Data:
    - Input images and segmentations are expected in NIfTI format (.nii.gz) with specific anatomical labels for bones and organs.
    - Strain masks and normal projection vectors are saved as NIfTI files with the appropriate transformations applied.

Outputs:
    - Strain masks and normal projection vectors are saved in the same directory as the input images, with filenames modified to reflect the computed strain information.

Usage:
    python make_physics_loss_masks.py --path {your_data_dir}AbdomenCTCT_reoriented/imagesTr
"""

import argparse
import glob
import os

import monai.transforms as MTransforms
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_dilation
from scipy.spatial import KDTree
from sklearn.decomposition import PCA


def compute_interface_normals(strain, num_neighbors=50):
    """_summary_

    Args:
        strain (_type_): _description_
        num_neighbors (int, optional):number of nearest neighbors to consider for plane fitting.
                                    Increase for better surface estimation. Defaults to 10.

    Returns:
        _type_: _description_
    """
    # Get surface voxel coordinates
    z, y, x = np.where(strain > 0)
    points = np.vstack((x, y, z)).T

    # Use KDTree for local neighborhood search
    tree = KDTree(points)

    normals = []
    if len(points) < 10:
        return [], []
    for point in points:
        # Find nearest neighbors
        dist, idx = tree.query(point, k=num_neighbors)
        idx = idx[dist < np.inf]

        neighbors = points[idx]

        # Use PCA to estimate the local surface plane
        pca = PCA(n_components=3)
        pca.fit(neighbors)
        normal = pca.components_[-1]  # The normal is the last principal component
        normal = np.round(normal, decimals=1)
        if normal[0] < 0:
            normal *= -1
        if normal[1] < 0:
            normal *= -1
        if normal[2] < 0:
            normal *= -1
        normal /= np.linalg.norm(normal)  # Normalize

        normals.append(normal)
    normals = np.array(normals)
    normals = np.round(normals, decimals=0)
    return points, torch.tensor(normals)


def strain_mask(segmentation, label1, label2, dilation_radius=1):
    """
    Computes a mask indicating where label1 and label2 are adjacent or nearly touching.

    Parameters:
    - segmentation: 3D numpy array (H, W, D) containing integer labels.
    - label1, label2: Integers corresponding to the labels of interest.
    - dilation_radius: Number of pixels/voxels to dilate (increases "touching" sensitivity).

    Returns:
    - strain_mask: Binary numpy array (H, W, D) where 1 indicates areas of contact/strain.
    """
    # Create binary masks for the two labels
    mask1 = (segmentation == label1).numpy()
    mask2 = (segmentation == label2).numpy()

    # Dilate both masks
    dilated_mask1 = torch.tensor(binary_dilation(mask1, iterations=dilation_radius))
    dilated_mask2 = torch.tensor(binary_dilation(mask2, iterations=dilation_radius))

    # Find intersection of dilated regions (strain areas)
    strain = dilated_mask1 & dilated_mask2  # .int()

    return strain  # Return as a binary mask (0 and 1)


rigid_movement = [
    (25, "sacrum"),
    (26, "vertebrae_S1"),
    (27, "vertebrae_L5"),
    (28, "vertebrae_L4"),
    (29, "vertebrae_L3"),
    (30, "vertebrae_L2"),
    (31, "vertebrae_L1"),
    (32, "vertebrae_T12"),
    (33, "vertebrae_T11"),
    (34, "vertebrae_T10"),
    (35, "vertebrae_T9"),
    (36, "vertebrae_T8"),
    (37, "vertebrae_T7"),
    (38, "vertebrae_T6"),
    (39, "vertebrae_T5"),
    (40, "vertebrae_T4"),
    (41, "vertebrae_T3"),
    (42, "vertebrae_T2"),
    (43, "vertebrae_T1"),
    (44, "vertebrae_C7"),
    (45, "vertebrae_C6"),
    (46, "vertebrae_C5"),
    (47, "vertebrae_C4"),
    (48, "vertebrae_C3"),
    (49, "vertebrae_C2"),
    (50, "vertebrae_C1"),
    (69, "humerus_left"),
    (70, "humerus_right"),
    (71, "scapula_left"),
    (72, "scapula_right"),
    (73, "clavicula_left"),
    (74, "clavicula_right"),
    (75, "femur_left"),
    (76, "femur_right"),
    (77, "hip_left"),
    (78, "hip_right"),
    (91, "skull"),
    (92, "rib_left_1"),
    (93, "rib_left_2"),
    (94, "rib_left_3"),
    (95, "rib_left_4"),
    (96, "rib_left_5"),
    (97, "rib_left_6"),
    (98, "rib_left_7"),
    (99, "rib_left_8"),
    (100, "rib_left_9"),
    (101, "rib_left_10"),
    (102, "rib_left_11"),
    (103, "rib_left_12"),
    (104, "rib_right_1"),
    (105, "rib_right_2"),
    (106, "rib_right_3"),
    (107, "rib_right_4"),
    (108, "rib_right_5"),
    (109, "rib_right_6"),
    (110, "rib_right_7"),
    (111, "rib_right_8"),
    (112, "rib_right_9"),
    (113, "rib_right_10"),
    (114, "rib_right_11"),
    (115, "rib_right_12"),
    (116, "sternum"),
    (117, "costal_cartilages"),
]

shear_stress_with_radius = [
    # Existing kidney and lung lobe contacts
    (2, "kidney_right", 24, "kidney_cyst_right", 3),
    (3, "kidney_left", 23, "kidney_cyst_left", 3),
    (10, "lung_upper_lobe_left", 11, "lung_lower_lobe_left", 5),
    (12, "lung_upper_lobe_right", 13, "lung_middle_lobe_right", 5),
    (13, "lung_middle_lobe_right", 14, "lung_lower_lobe_right", 5),
    # 🫁 LUNG - THORAX INTERACTIONS
    (
        10,
        "lung_upper_lobe_left",
        43,
        "vertebrae_T1",
        4,
    ),  # Left lung & upper thoracic spine
    (
        12,
        "lung_upper_lobe_right",
        43,
        "vertebrae_T1",
        4,
    ),  # Right lung & upper thoracic spine
    (
        11,
        "lung_lower_lobe_left",
        34,
        "vertebrae_T9",
        4,
    ),  # Left lung & lower thoracic spine
    (
        14,
        "lung_lower_lobe_right",
        34,
        "vertebrae_T9",
        4,
    ),  # Right lung & lower thoracic spine
    (10, "lung_upper_lobe_left", 92, "rib_left_1", 3),  # Left lung & first rib
    (12, "lung_upper_lobe_right", 104, "rib_right_1", 3),  # Right lung & first rib
    # 🏵️ LIVER - LUNG - SPLEEN INTERACTIONS
    (5, "liver", 14, "lung_lower_lobe_right", 5),  # Liver & right lung via diaphragm
    (5, "liver", 6, "stomach", 4),  # Liver & stomach (anterior contact)
    (5, "liver", 7, "pancreas", 3),  # Liver & pancreas
    (5, "liver", 8, "adrenal_gland_right", 3),  # Liver & right adrenal gland
    (5, "liver", 9, "adrenal_gland_left", 3),  # Liver & left adrenal gland
    (5, "liver", 18, "small_bowel", 2),  # Liver & small intestine (pressure contact)
    (5, "liver", 19, "duodenum", 3),  # Liver & duodenum
    (5, "liver", 20, "colon", 3),  # Liver & colon (hepatic flexure)
    (5, "liver", 1, "spleen", 5),  # Liver & spleen contact
    (1, "spleen", 3, "kidney_left", 4),  # Spleen & left kidney
    (1, "spleen", 6, "stomach", 3),  # Spleen & stomach
    (1, "spleen", 7, "pancreas", 3),  # Spleen & pancreas
    (1, "spleen", 92, "rib_left_10", 3),  # Spleen & rib cage
    (1, "spleen", 5, "diaphragm", 5),  # Spleen & diaphragm (pressure from lung motion)
    (
        1,
        "spleen",
        10,
        "lung_lower_lobe_left",
        5,
    ),  # Spleen & lung via diaphragm movement
    (1, "spleen", 92, "rib_left_9", 3),  # Spleen & 9th left rib
    (1, "spleen", 93, "rib_left_10", 3),  # Spleen & 10th left rib
    (1, "spleen", 94, "rib_left_11", 3),  # Spleen & 11th left rib
    (1, "spleen", 5, "liver", 5),  # Spleen & liver via intra-abdominal pressure
    (5, "liver", 11, "lung_lower_lobe_left", 6),  # Liver & lung (diaphragm interaction)
    # 🏥 ABDOMINAL STRUCTURES
    (7, "pancreas", 19, "duodenum", 3),  # Pancreas & duodenum
    (7, "pancreas", 20, "colon", 3),  # Pancreas & colon
    (7, "pancreas", 6, "stomach", 3),  # Pancreas & stomach
    (19, "duodenum", 20, "colon", 2),  # Duodenum & colon
    (18, "small_bowel", 19, "duodenum", 2),  # Small bowel & duodenum
    # ❤️ CARDIOVASCULAR SYSTEM
    (51, "heart", 52, "aorta", 4),
    (51, "heart", 53, "pulmonary_vein", 4),
    (52, "aorta", 54, "brachiocephalic_trunk", 4),
    (54, "brachiocephalic_trunk", 55, "subclavian_artery_right", 3),
    (54, "brachiocephalic_trunk", 57, "common_carotid_artery_right", 3),
    (55, "subclavian_artery_right", 56, "subclavian_artery_left", 3),
    (67, "iliac_vena_left", 68, "iliac_vena_right", 4),
    (65, "iliac_artery_left", 66, "iliac_artery_right", 4),
    (51, "heart", 16, "trachea", 4),  # Trachea-heart mechanical link via mediastinum
    (51, "heart", 5, "liver", 5),  # Heart and liver via diaphragm movement
    (16, "trachea", 52, "aorta", 3),  # Trachea-Aorta (pulsatile + respiratory stress)
    # # 🦴 HIP & LIMBS
    # (77, "hip_left", 75, "femur_left", 7),
    # (78, "hip_right", 76, "femur_right", 7),
    # (80, "gluteus_maximus_left", 82, "gluteus_medius_left", 5),
    # (81, "gluteus_maximus_right", 83, "gluteus_medius_right", 5),
    # (82, "gluteus_medius_left", 84, "gluteus_minimus_left", 4),
    # (83, "gluteus_medius_right", 85, "gluteus_minimus_right", 4),
    # (88, "iliopsoas_left", 77, "hip_left", 6),
    # (89, "iliopsoas_right", 78, "hip_right", 6),
    # # 🏋️ SHOULDER JOINT
    # (69, "humerus_left", 71, "scapula_left", 6),
    # (70, "humerus_right", 72, "scapula_right", 6)
    # I added these to account for "lower lobe" not "really" being lower...:
    (
        5,
        "liver",
        10,
        "lung_upper_lobe_left",
        6,
    ),  # <- (5, "liver", 11, "lung_lower_lobe_left", 6),
    (
        10,
        "lung_upper_lobe_left",
        34,
        "vertebrae_T9",
        4,
    ),  # <- (11, "lung_lower_lobe_left", 34, "vertebrae_T9", 4)
    (
        13,
        "lung_middle_lobe_right",
        34,
        "vertebrae_T9",
        4,
    ),  # <- (14, "lung_lower_lobe_right", 34, "vertebrae_T9", 4)
    (
        5,
        "liver",
        13,
        "lung_middle_lobe_right",
        5,
    ),  # <- (5, "liver", 14, "lung_lower_lobe_right", 5)
    (
        1,
        "spleen",
        11,
        "lung_lower_lobe_left",
        5,
    ),  # <- (1, "spleen", 10, "lung_lower_lobe_left", 5)
]


#####
def main(args):
    tr_path = args.path
    ts_path = args.path.replace("imagesTr", "imagesTs")
    pathsTR = glob.glob(f"{tr_path}/*.nii.gz")
    pathsTS = glob.glob(f"{ts_path}/*.nii.gz")
    paths_all = pathsTR + pathsTS
    files_all = []
    for path in paths_all:
        name = os.path.basename(path).split(".")[0]
        files_all.append(
            {
                "total": path.replace("imagesTr", "segmentationsTr/total/").replace(
                    "imagesTs", "segmentationsTs/total/"
                ),
            }
        )

    keys = ["total"]
    list_transforms = [MTransforms.LoadImaged(keys=keys, ensure_channel_first=True)]
    loader = MTransforms.Compose(list_transforms)

    for data_dict in files_all:
        data = loader([data_dict])[0]
        total = data["total"].squeeze()
        loss_mask = torch.zeros_like(total)
        mask_strain_directions = torch.zeros(
            3, total.shape[0], total.shape[1], total.shape[2]
        )
        # 0 means DetJac loss / 1 means rigidity loss / 2 means shearing loss
        for label, name in rigid_movement:
            loss_mask[total == label] = 1
        for label1, name1, label2, name2, radius in shear_stress_with_radius:
            mask = strain_mask(total, label1, label2, dilation_radius=radius // 2)  # 5)

            # # Get normals:
            points, normals = compute_interface_normals(
                mask.squeeze(), num_neighbors=40
            )
            for num, (x, y, z) in enumerate(points):
                mask_strain_directions[:, z, y, x] = normals[num, :]

            if mask.int().max() == 0:
                print(f" None found between {name1} and {name2}")
            else:
                print(
                    f" ---- found {mask.int().sum().item()} between {name1} and {name2}"
                )
                loss_mask[mask] = 2

        # Save masks :
        affine = total.meta["affine"]
        save_name = total.meta["filename_or_obj"].replace("total", "strain_mask")
        save_folder = os.path.dirname(save_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        nib.save(nib.Nifti1Image(loss_mask.squeeze().cpu().numpy(), affine), save_name)
        # Save projection vectors :
        affine = total.meta["affine"]
        save_name = total.meta["filename_or_obj"].replace("total", "projection_vectors")
        save_folder = os.path.dirname(save_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        nib.save(
            nib.Nifti1Image(
                mask_strain_directions.squeeze().permute(1, 2, 3, 0).cpu().numpy(),
                affine,
            ),
            save_name,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=os.path.abspath,
        default="Datasets/AbdomenCTCT_reoriented/imagesTr",
        help="Path toLearn2Reg train data locations",
    )
    args = parser.parse_args()

    main(args)
