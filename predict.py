import os
import time
import pandas as pd
import numpy as np
import json
from helper_functions import combine_features, load_trained_models, average_prediction, weighted_average_prediction

def read_data(settings):
    de_train = pd.read_parquet(settings["TRAIN_RAW_DATA_PATH"])
    id_map = pd.read_csv(settings["TEST_RAW_DATA_PATH"])
    sample_submission = pd.read_csv(settings["SAMPLE_SUBMISSION_PATH"], index_col='id')
    return de_train, id_map, sample_submission

if __name__ == "__main__":
    ## Read settings and config files
    with open("./SETTINGS.json") as file:
        settings = json.load(file)
    with open("./config/test_config.json") as file:
        test_config = json.load(file)
        
    ## Read train, test and sample submission data # train data is needed for columns
    print("\nReading data...")
    de_train, id_map, sample_submission = read_data(settings)
    
    ## Build input features
    mean_cell_type = pd.read_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}mean_cell_type.csv')
    std_cell_type = pd.read_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}std_cell_type.csv')
    mean_sm_name = pd.read_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}mean_sm_name.csv')
    std_sm_name = pd.read_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}std_sm_name.csv')
    quantiles_df = pd.read_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}quantiles_cell_type.csv')
    test_chem_feat = np.load(f'{settings["TRAIN_DATA_AUG_DIR"]}chemberta_test.npy')
    test_chem_feat_mean = np.load(f'{settings["TRAIN_DATA_AUG_DIR"]}chemberta_test_mean.npy')
    one_hot_test = pd.DataFrame(np.load(f'{settings["TRAIN_DATA_AUG_DIR"]}one_hot_test.npy'))
    
    test_vec = combine_features([mean_cell_type, std_cell_type, mean_sm_name, std_sm_name],\
                [test_chem_feat, test_chem_feat_mean], id_map, one_hot_test)
    test_vec_light = combine_features([mean_cell_type,mean_sm_name],\
                    [test_chem_feat, test_chem_feat_mean], id_map, one_hot_test)
    test_vec_heavy = combine_features([quantiles_df,mean_cell_type,mean_sm_name],\
                    [test_chem_feat,test_chem_feat_mean], id_map, one_hot_test, quantiles_df)
    
    ## Load trained models
    print("\nLoading trained models...")
    trained_models = load_trained_models(path=f'{settings["MODEL_DIR"]}')
    fold_weights = test_config["FOLD_COEFS"] if test_config["KF_N_SPLITS"] == 5 else [1.0/test_config["KF_N_SPLITS"]]*test_config["KF_N_SPLITS"]
    ## Start predictions
    print("\nStarting predictions...")
    t0 = time.time()
    pred1 = average_prediction(test_vec_light, trained_models['light'])
    pred2 = weighted_average_prediction(test_vec_light, trained_models['light'],\
                                        model_wise=test_config["MODEL_COEFS"], fold_wise=fold_weights)
    pred3 = average_prediction(test_vec, trained_models['initial'])
    pred4 = weighted_average_prediction(test_vec, trained_models['initial'],\
                                        model_wise=test_config["MODEL_COEFS"], fold_wise=fold_weights)
    
    pred5 = average_prediction(test_vec_heavy, trained_models['heavy'])
    pred6 = weighted_average_prediction(test_vec_heavy, trained_models['heavy'],\
                                    model_wise=test_config["MODEL_COEFS"], fold_wise=fold_weights)
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
    if not os.path.exists(settings["SUBMISSION_DIR"]):
        os.mkdir(settings["SUBMISSION_DIR"])
    df_sub.to_csv(f'{settings["SUBMISSION_DIR"]}submission.csv')
    print("\nDone.")