# Code supporting paper

Download data : (or manually: https://cloud.imi.uni-luebeck.de/s/32WaSRaTnFk2JeT)
1) mkdir Datasets/
2) wget -P Datasets/ https://cloud.imi.uni-luebeck.de/s/32WaSRaTnFk2JeT/download/AbdomenCTCT.zip
2) unzip Datasets/AbdomenCTCT.zip -d Datasets/
3) rm -r Datasets/AbdomenCTCT.zip

Download dependendecies:
1) conda create --name biomechanical_DLIR python=3.9.16
2) conda activate biomechanical_DLIR


For AbdomenCTCT dataset :
<!-- Download dependendecies: -->

conda create --name abdomenCTCT python=3.10
pip install numpy==1.26.4
pip install nibabel==5.0.1
pip install TotalSegmentator==2.7.0

conda create --name biomechanical_dlir python=3.9.16
pip install wandb==0.15.0
pip install loguru==0.7.3
pip install monai==1.4.0