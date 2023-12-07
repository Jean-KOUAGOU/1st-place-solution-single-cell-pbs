import os
import argparse
import time
import pandas as pd
import numpy as np
from helper_functions import read_config, combine_features, load_trained_models, average_prediction, weighted_average_prediction

def read_data(config):
    de_train = pd.read_parquet(CONFIG.train_data_path)
    id_map = pd.read_csv(CONFIG.test_data_path)
    sample_submission = pd.read_csv(CONFIG.sample_submission_path, index_col='id')
    return de_train, id_map, sample_submission

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kf_n_splits", type=int, default=5, help="The number of epochs")
    args = parser.parse_args()
    
    ## Read config, train and test data # only train data columns are needed
    print("\nReading data...")
    CONFIG  = read_config("./config/config_test.json")
    de_train, id_map, sample_submission = read_data(CONFIG)
    
    ## Build input features
    mean_cell_type = pd.read_csv(CONFIG.mean_cell_type)
    std_cell_type = pd.read_csv(CONFIG.std_cell_type)
    mean_sm_name = pd.read_csv(CONFIG.mean_sm_name)
    std_sm_name = pd.read_csv(CONFIG.std_sm_name)
    quantiles_df = pd.read_csv(CONFIG.quantiles_cell_type)
    test_chem_feat = np.load(CONFIG.chemberta_test)
    test_chem_feat_mean = np.load(CONFIG.chemberta_test_mean)
    one_hot_test = pd.DataFrame(np.load(CONFIG.one_hot_test))
    
    test_vec = combine_features([mean_cell_type, std_cell_type, mean_sm_name, std_sm_name],\
                [test_chem_feat, test_chem_feat_mean], id_map, one_hot_test)
    test_vec_light = combine_features([mean_cell_type,mean_sm_name],\
                    [test_chem_feat, test_chem_feat_mean], id_map, one_hot_test)
    test_vec_heavy = combine_features([quantiles_df,mean_cell_type,mean_sm_name],\
                    [test_chem_feat,test_chem_feat_mean], id_map, one_hot_test, quantiles_df)
    
    ## Load trained models
    print("\nLoading trained models...")
    trained_models = load_trained_models(path=CONFIG.trained_models_path)
    fold_weights = [0.25, 0.15, 0.2, 0.15, 0.25] if args.kf_n_splits == 5 else [1.0/args.kf_n_splits]*args.kf_n_splits
    ## Start predictions
    print("\nStarting predictions...")
    t0 = time.time()
    pred1 = average_prediction(test_vec_light, trained_models['light'])
    pred2 = weighted_average_prediction(test_vec_light, trained_models['light'],\
                                        model_wise=[0.29, 0.33, 0.38], fold_wise=fold_weights)
    pred3 = average_prediction(test_vec, trained_models['initial'])
    pred4 = weighted_average_prediction(test_vec, trained_models['initial'],\
                                        model_wise=[0.29, 0.33, 0.38], fold_wise=fold_weights)
    
    pred5 = average_prediction(test_vec_heavy, trained_models['heavy'])
    pred6 = weighted_average_prediction(test_vec_heavy, trained_models['heavy'],\
                                    model_wise=[0.29, 0.33, 0.38], fold_wise=fold_weights)
    t1 = time.time()
    print("Prediction time: ", t1-t0, " seconds")
    print("\nEnsembling predictions and writing to file...")
    col = list(de_train.columns[5:])
    submission = sample_submission.copy()
    
    submission[col] = 0.23*pred1 + 0.15*pred2 + 0.18*pred3 + 0.15*pred4 + 0.15*pred5 + 0.14*pred6
    df1 = submission.copy()
    
    submission[col] = 0.13*pred1 + 0.15*pred2 + 0.23*pred3 + 0.15*pred4 + 0.20*pred5 + 0.14*pred6
    df2 = submission.copy()
    
    submission[col] = 0.17*pred1 + 0.16*pred2 + 0.17*pred3 + 0.16*pred4 + 0.18*pred5 + 0.16*pred6
    df3 = submission.copy()
    df_sub = 0.34*df1 + 0.33*df2 + 0.33*df3 # Final ensembling
    if not os.path.exists("./submissions"):
        os.mkdir("./submissions")
    df_sub.to_csv('./submissions/submission.csv')
    print("\nDone.")