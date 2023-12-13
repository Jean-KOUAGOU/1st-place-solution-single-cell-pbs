1. python prepare_data.py
- reads training data from TRAIN_RAW_DATA_DIR (specified in SETTINGS.json)
- runs all preprocessing steps
- saves the prepared data in TRAIN_DATA_AUG_DIR (specified in SETTINGS.json)
2. python train.py
- reads training data from TRAIN_RAW_DATA_PATH and TRAIN_DATA_AUG_DIR (specified in SETTINGS.json)
- trains the models
- saves the trained models to MODEL_DIR (specified in SETTINGS.json)
3. python predict.py
- reads test data from TEST_RAW_DATA_DIR and TRAIN_DATA_AUG_DIR, e.g., mean, std, quantiles per cell_type and sm_name (specified in SETTINGS.json)
- loads the trained models from MODEL_DIR (specified in SETTINGS.json)
- uses the trained models to make predictions on new samples
- saves the predictions to SUBMISSION_DIR (specified in SETTINGS.json)
