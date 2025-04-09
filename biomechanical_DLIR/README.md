# Scripts to train and evaluate models described in paper

python main.py -m configs/simulated_configs_grid/RigidityDet/MSE_strainDet0_99.json 
python main.py -m configs/simulatedShear_configs_grid/StrainDetShear/MSE_rigidityDetShear0_99.json 

Troubleshooting:
"_pickle.PicklingError: Can't pickle..." ->  Change num_workers in main.py to 0 