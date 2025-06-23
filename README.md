# Biomechanical Constraints Assimilation in Deep-Learning Image Registration
Supporting code for reproducing [Biomechanical Constraints Assimilation in Deep-Learning Image Registration: Application to sliding and locally rigid deformations](https://arxiv.org/abs/2504.05444)

## Install dependencies
- ```conda create --name biomechanical_DLIR python=3.10```
- ```conda activate biomechanical_DLIR```   
- ```pip install -r requirements.txt```

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

You should now have the complete dataset under _Datasets/AbdomenCTCT_reoriented/_ including a _/segmentationsTr_ folder.

## Training and Evaluation Script
Model architectures, training orchestration and dataset configurations are handled through a json file.
[The main training script](biomechanical_DLIR/main.py) then executes the desired training. 
We provide all the [configurations](biomechanical_DLIR/configs) used for our hyper-parameter grid-search, specifically:
- [configurations](biomechanical_DLIR/configs/simulated_configs_grid/) -> scripts for synthetic rotation dataset.
- [configurations](biomechanical_DLIR/configs/simulatedShear_configs_grid) -> scripts for synthetic shearing dataset.
- [configurations](biomechanical_DLIR/configs/abdomenCTCT_grid/) -> scripts for AbdomenCTCT dataset.

Model training is then simply achieved by : 
```python biomechanical_DLIR/main.py -m biomechanical_DLIR/configs/simulated_configs_grid/RigidityDet/MSE_strainDet0_99.json ```

Models and logs will be saved under a new directory named results/ .

Finally, models can be evaluated with [eval.py](biomechanical_DLIR/eval.py) (or for full resolution on the AbdomenCTCT dataset: [eval_full_res.py](biomechanical_DLIR/eval_full_res.py)).

For more information:
-> ```python biomechanical_DLIR/main.py -h ```
-> ```python biomechanical_DLIR/eval.py -h ```
-> ```python biomechanical_DLIR/eval_full_res.py -h ```

## Custom Dataset or Models
Implementing a custom dataset requires adding a [custom dataset class](biomechanical_DLIR/src/datasets/) , then calling that dataset in the json object.
Likewise, [custom models](biomechanical_DLIR/src/models/) can be added.


# 📝 Citation
```
@misc{kheil2025biomechanicalconstraintsassimilationdeeplearning,
      title={Biomechanical Constraints Assimilation in Deep-Learning Image Registration: Application to sliding and locally rigid deformations}, 
      author={Ziad Kheil and Soleakhena Ken and Laurent Risser},
      year={2025},
      eprint={2504.05444},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.05444}, 
}
```

