# 1st-place-solution-single-cell-pbs
This repository implements the winning solution for the single cell perturbation problems

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

2. Add the training data in parquet format, e.g., `de_train.parquet` as in the competition. Make sure to edit `config/config_train.json` by specifying the correct training data path. The same path should be entered in `config/config_test` as it is needed to read columns at inference time.

3. Also add the test data and a sample submission file (both should be csv files) in the same directory `data/` and edit `config/config_test.json` accordingly. Note that paths to trained models and to the training data should be specified in `config/config_test.json`.

4. Run `python prepare_data.py` to complete all required preprocessing steps

## Training
Make sure to locate in the main directory
- Run `python train.py` to train models. This will automatically create a directory call `trained_models` and store the trained models. One can specify the number of folds in the K-fold cross-validation scheme using `python train.py --kf_n_splis`. Similarly, the number of epochs can be specified using `--epochs`
- Pretrained models can also be downloaded, see link on Kaggle to avoid training.

## Predicting
Check that there is a non-empty directory named `trained_models` and that its path is specified in `config/config_test.json`
- Run `python predict.py` to predict on the test data whose path is specified in `config/config_test.json`. This will automatically create a directory `submissions` and store predictions in `submissions/submission.csv`



