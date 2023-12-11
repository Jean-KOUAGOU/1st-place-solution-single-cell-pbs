from argparse import Namespace
import pandas as pd
import numpy as np
import json
from helper_functions import seed_everything, combine_features, train_validate

if __name__ == "__main__":
    # Read settings and config files
    with open("./SETTINGS.json") as file:
        settings = json.load(file)
    with open("./config/train_config.json") as file:
        train_config = json.load(file)
        
    print("\nRead data and build features...")
    de_train = pd.read_parquet(settings["TRAIN_RAW_DATA_PATH"])
    xlist  = ['cell_type','sm_name']
    ylist = ['cell_type','sm_name','sm_lincs_id','SMILES','control']
    one_hot_train = pd.DataFrame(np.load(f'{settings["TRAIN_DATA_AUG_DIR"]}one_hot_train.npy'))
    y = de_train.drop(columns=ylist)
    mean_cell_type = pd.read_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}mean_cell_type.csv')
    std_cell_type = pd.read_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}std_cell_type.csv')
    mean_sm_name = pd.read_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}mean_sm_name.csv')
    std_sm_name = pd.read_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}std_sm_name.csv')
    quantiles_df = pd.read_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}quantiles_cell_type.csv')
    train_chem_feat = np.load(f'{settings["TRAIN_DATA_AUG_DIR"]}chemberta_train.npy')
    train_chem_feat_mean = np.load(f'{settings["TRAIN_DATA_AUG_DIR"]}chemberta_train_mean.npy')
    X_vec = combine_features([mean_cell_type, std_cell_type, mean_sm_name, std_sm_name],\
                [train_chem_feat, train_chem_feat_mean], de_train, one_hot_train)
    X_vec_light = combine_features([mean_cell_type,mean_sm_name],\
                    [train_chem_feat, train_chem_feat_mean], de_train, one_hot_train)
    X_vec_heavy = combine_features([quantiles_df,mean_cell_type,mean_sm_name],\
                    [train_chem_feat,train_chem_feat_mean], de_train, one_hot_train, quantiles_df)
    ## Start training
    cell_types_sm_names = de_train[['cell_type', 'sm_name']]
    print("\nTraining starting...")
    train_validate(X_vec, X_vec_light, X_vec_heavy, y, cell_types_sm_names, train_config)
    print("\nDone.")
    
    
    
