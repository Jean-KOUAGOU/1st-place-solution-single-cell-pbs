import argparse
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, default='predict', choices=['train_and_predict', 'predict'], help="Whether to train before predicting or predict directly")
    args = parser.parse_args()
    # Preprocess data, apply data augmentation techniques such as computing mean, std, quantiles of the target per cell type and sm name
    os.system(f'python prepare_data.py')
    if args.action == 'train_and_predict':
        # First train
        os.system(f'python train.py')
        # Then predict
        os.system(f'python predict.py')
    else:
        # Only predict using trained models in SETTINGS: MODEL_DIR
        os.system(f'python predict.py')