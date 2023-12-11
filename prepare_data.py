from argparse import Namespace
import os
import json
import pandas as pd
from helper_functions import seed_everything, one_hot_encode, save_ChemBERTa_features

if __name__ == "__main__":
    ## Seed for reproducibility
    seed_everything()
    with open("./SETTINGS.json") as file:
        settings = json.load(file)
    ## Read data
    print("\nProcessing...")
    de_train = pd.read_parquet(settings["TRAIN_RAW_DATA_PATH"])
    id_map = pd.read_csv(settings["TEST_RAW_DATA_PATH"])
    ## Create data augmentation
    de_cell_type = de_train.iloc[:, [0] + list(range(5, de_train.shape[1]))]
    de_sm_name = de_train.iloc[:, [1] + list(range(5, de_train.shape[1]))]
    mean_cell_type = de_cell_type.groupby('cell_type').mean().reset_index()
    mean_sm_name = de_sm_name.groupby('sm_name').mean().reset_index()
    std_cell_type = de_cell_type.groupby('cell_type').std().reset_index()
    std_sm_name = de_sm_name.groupby('sm_name').std().reset_index()
    cell_types = de_cell_type.groupby('cell_type').quantile(0.1).reset_index()['cell_type']
    quantiles_cell_type = pd.concat([pd.DataFrame(cell_types)]+[de_cell_type.groupby('cell_type')[col]\
.quantile([0.25, 0.50, 0.75], interpolation='linear').unstack().reset_index(drop=True) for col in list(de_train.columns)[5:]], axis=1)
    ## Save data augmentation features
    if not os.path.exists(settings["TRAIN_DATA_AUG_DIR"]):
        os.mkdir(settings["TRAIN_DATA_AUG_DIR"])
    mean_cell_type.to_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}mean_cell_type.csv', index=False)
    std_cell_type.to_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}std_cell_type.csv', index=False)
    mean_sm_name.to_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}mean_sm_name.csv', index=False)
    std_sm_name.to_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}std_sm_name.csv', index=False)
    quantiles_cell_type.to_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}quantiles_cell_type.csv', index=False)
    ## Create one hot encoding features
    one_hot_encode(de_train[["cell_type", "sm_name"]], id_map[["cell_type", "sm_name"]], out_dir=settings["TRAIN_DATA_AUG_DIR"])
    ## Prepare ChemBERTa features
    save_ChemBERTa_features(de_train["SMILES"].tolist(), out_dir=settings["TRAIN_DATA_AUG_DIR"], on_train_data=True)
    sm_name2smiles = {smname:smiles for smname, smiles in zip(de_train['sm_name'], de_train['SMILES'])}
    test_smiles = list(map(sm_name2smiles.get, id_map['sm_name'].values))
    save_ChemBERTa_features(test_smiles, out_dir=settings["TRAIN_DATA_AUG_DIR"], on_train_data=False)
    print("### Done.")