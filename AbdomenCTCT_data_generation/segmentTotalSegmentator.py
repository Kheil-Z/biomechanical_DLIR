"""
    Script to segment reoriented AbdomenCTCT data using totla segmentator
    Used specifically for Learn2Reg AbdomenCTCT dataset, to segment using TotalSegmentator.

Usage:
    python segmentTotalSegmentator.py --path {your_data_dir}/AbdomenCTCT_reoriented/imagesTr
"""

import argparse
import glob
import os


def main(args):
    tr_path = args.path
    ts_path = args.path.replace("imagesTr", "imagesTs")
    pathsTR = glob.glob(f"{tr_path}/*.nii.gz")
    pathsTS = glob.glob(f"{ts_path}/*.nii.gz")
    paths_all = pathsTR + pathsTS

    # TotalSegmentator info:
    task_list = [
        "total",
        "body",
        "lung_nodules",
    ]  # ["total","lung_vessels","body","liver_vessels","lung_nodules"]
    current_total = 0
    for path_num, p in enumerate(paths_all):
        for task_num, task in enumerate(task_list):
            current_total += 1
            out_path = p.replace("imagesTr/", f"segmentationsTr/{task}/").replace(
                "imagesTs/", f"segmentationsTs/{task}/"
            )
            # # os.system(f"TotalSegmentator -i {p} -o {out_path} --task {task} --preview --device gpu --ml")
            os.system(
                f"TotalSegmentator -i {p} -o {out_path} --task {task}  --device gpu --ml"
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
