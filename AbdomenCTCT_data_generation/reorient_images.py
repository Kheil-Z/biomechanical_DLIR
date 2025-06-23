"""
    Script to reorient nifti images (from "RAS" to "LAS" orientation).
    Used specifically for Learn2Reg AbdomenCTCT dataset, to segment using TotalSegmentator.

Usage:
    python reorient_images.py --path {your_data_dir}/AbdomenCTCT/imagesTr
"""

import argparse
import os

import nibabel as nib


def reorient_to_rpi(nifti_img):
    # Get the current orientation
    current_ornt = nib.orientations.axcodes2ornt(
        nib.orientations.aff2axcodes(nifti_img.affine)
    )

    # Define the desired RPI orientation
    target_ornt = nib.orientations.axcodes2ornt(("L", "A", "S"))

    # Get transformation to RPI
    transform = nib.orientations.ornt_transform(current_ornt, target_ornt)

    # Apply the transformation
    # reoriented_data = nib.orientations.apply_orientation(nifti_img.get_fdata(), transform)
    new_affine = nib.orientations.inv_ornt_aff(transform, nifti_img.shape)
    old_affine = nifti_img.affine
    new_affine = old_affine @ new_affine
    new_affine[:3, 3] = 0

    # return nib.Nifti1Image(reoriented_data, new_affine)#, nifti_img.header)
    return nib.Nifti1Image(nifti_img.get_fdata(), new_affine)  # , nifti_img.header)


def convert_nifti_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)

            img = nib.load(input_path)
            current_orientation = nib.orientations.aff2axcodes(img.affine)
            if current_orientation == ("R", "A", "S"):
                print(f"Converting {file} from RAS to LAS...")
                img_rpi = reorient_to_rpi(img)
                nib.save(img_rpi, output_path)
            else:
                print(
                    f"------- Skipping {file}, not in LPI orientation. ({current_orientation})"
                )


def main(args):
    input_dir = args.path
    output_dir = input_dir.replace("AbdomenCTCT", "AbdomenCTCT_reoriented")
    convert_nifti_directory(input_dir, output_dir)

    convert_nifti_directory(
        input_dir.replace("imagesTr", "imagesTs"),
        output_dir.replace("imagesTr", "imagesTs"),
    )
    convert_nifti_directory(
        input_dir.replace("imagesTr", "labelsTr"),
        output_dir.replace("imagesTr", "labelsTr"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=os.path.abspath,
        default="Datasets/AbdomenCTCT/imagesTr",
        help="Path toLearn2Reg train data locations",
    )
    args = parser.parse_args()

    main(args)
