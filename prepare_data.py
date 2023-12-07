from argparse import Namespace
import os
import json
import pandas as pd
from helper_functions import seed_everything, one_hot_encode, save_ChemBERTa_features

def save_data_aug_path_to_config(config, var_name, path, on_train=False):
    setattr(config, var_name, path)
    out_file = "./config/config_train.json" if on_train else "./config/config_test.json"
    with open(out_file, "w") as file:
        json.dump(vars(config), file)
    return None

if __name__ == "__main__":
    ## Seed for reproducibility
    seed_everything()
    with open("./config/config_train.json") as file:
        config_train = json.load(file)
        config_train = Namespace(**config_train)
    with open("./config/config_test.json") as file:
        config_test = json.load(file)
        config_test = Namespace(**config_test)
    ## Read data
    print("\nProcessing...")
    de_train = pd.read_parquet(config_train.train_data_path)
    id_map = pd.read_csv(config_test.test_data_path)
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
    if not os.path.exists("./prepared_data"):
        os.mkdir("./prepared_data")
    mean_cell_type.to_csv("./prepared_data/mean_cell_type.csv", index=False)
    std_cell_type.to_csv("./prepared_data/std_cell_type.csv", index=False)
    mean_sm_name.to_csv("./prepared_data/mean_sm_name.csv", index=False)
    std_sm_name.to_csv("./prepared_data/std_sm_name.csv", index=False)
    quantiles_cell_type.to_csv("./prepared_data/quantiles_cell_type.csv", index=False)
    for var_name, path in zip(["mean_cell_type", "std_cell_type", "mean_sm_name", "std_sm_name", "quantiles_cell_type"],
                              ["./prepared_data/mean_cell_type.csv", "./prepared_data/std_cell_type.csv",\
                               "./prepared_data/mean_sm_name.csv","./prepared_data/std_sm_name.csv", "./prepared_data/quantiles_cell_type.csv"]):
        save_data_aug_path_to_config(config_train, var_name, path, on_train=True)
        save_data_aug_path_to_config(config_test, var_name, path, on_train=False)
    ## Create one hot encoding features
    one_hot_encode(de_train[["cell_type", "sm_name"]], id_map[["cell_type", "sm_name"]], out_dir="./prepared_data", config_dir="./config")
    ## Prepare ChemBERTa features
    save_ChemBERTa_features(de_train["SMILES"].tolist(), out_dir="./prepared_data", config_dir="./config", on_train_data=True)
    sm_name2smiles = {smname:smiles for smname, smiles in zip(de_train['sm_name'], de_train['SMILES'])}
    test_smiles = list(map(sm_name2smiles.get, id_map['sm_name'].values))
    save_ChemBERTa_features(test_smiles, out_dir="./prepared_data", config_dir="./config", on_train_data=False)
    print("### Done.")