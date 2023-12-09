import argparse
from argparse import Namespace
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold as KF
from helper_functions import seed_everything, combine_features, read_config, train_validate


def read_data(config):
    de_train = pd.read_parquet(CONFIG.train_data_path)
    return de_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=250, help="The number of epochs")
    parser.add_argument("--kf_n_splits", type=int, default=5, help="The number of folds")
    args = parser.parse_args()
    CONFIG = read_config()
    print("\nRead data and build features...")
    de_train = read_data(CONFIG)
    xlist  = ['cell_type','sm_name']
    ylist = ['cell_type','sm_name','sm_lincs_id','SMILES','control']
    one_hot_train = pd.DataFrame(np.load(CONFIG.one_hot_train))
    y = de_train.drop(columns=ylist)
    mean_cell_type = pd.read_csv(CONFIG.mean_cell_type)
    std_cell_type = pd.read_csv(CONFIG.std_cell_type)
    mean_sm_name = pd.read_csv(CONFIG.mean_sm_name)
    std_sm_name = pd.read_csv(CONFIG.std_sm_name)
    quantiles_df = pd.read_csv(CONFIG.quantiles_cell_type)
    train_chem_feat = np.load(CONFIG.chemberta_train)
    train_chem_feat_mean = np.load(CONFIG.chemberta_train_mean)
    
    X_vec = combine_features([mean_cell_type, std_cell_type, mean_sm_name, std_sm_name],\
                [train_chem_feat, train_chem_feat_mean], de_train, one_hot_train)
    X_vec_light = combine_features([mean_cell_type,mean_sm_name],\
                    [train_chem_feat, train_chem_feat_mean], de_train, one_hot_train)
    X_vec_heavy = combine_features([quantiles_df,mean_cell_type,mean_sm_name],\
                    [train_chem_feat,train_chem_feat_mean], de_train, one_hot_train, quantiles_df)
    ## KFold cross validation
    splits = args.kf_n_splits
    kf_cv = KF(n_splits=splits, shuffle=True, random_state=42)
    ## Start training
    cell_types_sm_names = de_train[['cell_type', 'sm_name']]
    print("\nTraining starting...")
    train_validate(X_vec, X_vec_light, X_vec_heavy, y, kf_cv, cell_types_sm_names, epochs=args.epochs)
    print("\nDone.")
    
    
    
