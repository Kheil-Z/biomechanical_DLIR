# Scripts to generate required labels for training on Learn2Reg AbdomenCTCT data

## Download Learn2Reg AbdomenCTCT data and preprocess (skip to [Training Script](#training-script))
Part of our results were obtained on the [Learn2Reg AbdomenCTCT](https://learn2reg.grand-challenge.org/Datasets/) dataset.
1) Download data : (or manually [here](https://cloud.imi.uni-luebeck.de/s/32WaSRaTnFk2JeT) )
- ```mkdir Datasets/ ```
- ```wget -P Datasets/ https://cloud.imi.uni-luebeck.de/s/32WaSRaTnFk2JeT/download/AbdomenCTCT.zip ``` or ```curl ... ```
- ```unzip Datasets/AbdomenCTCT.zip -d Datasets/ ```
- ```rm -r Datasets/AbdomenCTCT.zip```

2) Generate regularization masks indicating which loss (rigidity,strain,Jacobian) to use for every voxel, as well as the sliding motion directions:
- ```python AbdomenCTCT_data_generation/reorient_images.py ```
- ```python AbdomenCTCT_data_generation/segmentTotalSegmentator.py ``` -> _Note that GPU usage is highly recommended.._
- ``` python AbdomenCTCT_data_generation/make_physics_loss_masks.py ```
