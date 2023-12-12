# 1st-place-solution-single-cell-pbs
This repository implements the 1st place solution for the single cell perturbations problem [open-problems-single-cell-perturbations](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/overview)

## General Methodology 
1. Input Features 
- Use one hot encoding of cell_type/sm_names
- Add the mean, standard deviation, and (25%, 50%, 75%) percentiles of target values (differential expressions) per cell_type and sm_name.
2. Model Architectures
- Use LSTM, GRU, and 1d-CNN architectures (see `models.py`). 
3. Loss Functions and Optimizer
Use MSE, MAE, BCE, and LogCosh (see `helper_classes.py`), and the Adam optimizer to train the models in a 5-fold cross validation setting.
4. Hyperparameters
- 250 epochs, lr 0.001 for LSTM and 1d-CNN, and 0.0003 for GRU.
- Use gradient norm clip value of 1.0 during training
- Batch size 16
5. Predictions
- Use weighted ensemble prediction; fold-wise, use the coefficients [0.25, 0.15, 0.2, 0.15, 0.25], and model-wise use [0.29, 0.33, 0.38]


## Installation
Make sure Anaconda3 is installed and execute the following:

1. Clone this repository `git clone https://github.com/Jean-KOUAGOU/1st-place-solution-single-cell-pbs.git`

2. First create and activate a conda environement `conda create -n single_cell_env python==3.9.0 --y && conda activate single_cell_env`

3. Install all required packages in the environment `pip install -r requirements.txt`

## Dependencies
1. python 3.9.0
2. pandas 2.1.3
3. pyarrow 14.0.1
4. tqdm 4.66.1
5. scikit-learn 1.3.2
6. torch 2.1.1
7. transformers 4.35.2
8. matplotlib 3.8.2

## Hardware:
- Ubuntu 20.04.6 LTS (Kaggle) AMD EPYC 7B12 CPU @ 2.25GHz (4 CPUs) 30GB RAM, 1xTesla GPU P100 16 GB (Kaggle), 73 GB disc
- Also tested on Debian GNU/Linux 11, 1xNvidia GPU rtx 3090 24 GB, 252 GB RAM, 500 GB disc


## Preprocessing
1. Create a folder called `data/` in the main directory

2. Add the training data in parquet format, e.g., `de_train.parquet` as in the competition and check that its path is correct in `SETTINGS.json`

3. Also add the test data and a sample submission file (both should be csv files) in the same directory `data/` and check `SETTINGS.json` for path correctness

4. Run `python prepare_data.py` to complete all required preprocessing steps

## Training
Make sure to locate at the top level of this Github repository
- Run `python train.py` to train models. This will automatically create a directory call `trained_models` and store the trained models.
- Pretrained models can also be downloaded, see link on Kaggle to avoid training.

## Predicting
Check that there is a non-empty directory named `trained_models` and that its path is specified in `SETTINGS.json` under `MODEL_DIR`
- Run `python predict.py` to predict on the test data whose path is specified in `SETTINGS.json`. This will automatically create an output directory sepcified in `SETTINGS.json`and store predictions in a file named `submission.csv`

## Reproduction
1. Create a directory `data` in this Github repository
2. Add de_train.parquet, id_map.csv, and sample_submission.csv into the directory `data`
3. If necessary, edit SETTINGS.json by specifying the correct paths
4. Make sure your machine has at least 16GB RAM
5. Execute `./build.sh` to build a docker image
6. If you would like to predict with pretrained models:
- Download the trained models from Kaggle, and place them under a folder named `trained_models` at the top level of this Github repository
- Execute `./run.sh predict` to run the container and directly predict using the trained models
7. Execute `./run.sh train_and_predict` to train new models and predict. If the objective is not to reproduce the results, you can also change configurations in `config` such as learning rate, epochs, etc.
Note: If you encounter an error in 6. and 7., there is probably a conflicting container name, e.g., you have executed `./run.sh` several times. In that case, delete the container id, and retry.



