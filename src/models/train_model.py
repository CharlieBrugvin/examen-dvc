"""Entraînement du modèle.

En utilisant les paramètres retrouvés à travers le GridSearch, on entraînera le modèle en sauvegardant le modèle entraîné dans le dossier models.
"""
import logging
from pathlib import Path
import pickle
import joblib

import numpy as np
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(level=logging.INFO)


INPUT_PARAMETERS_FILE = Path("models/best_parameters.pkl")
INPUT_DATA_FOLDER = Path("data/processed_data")
SEED = 42
OUTPUT_MODEL_FILE = Path("models/regressor_model.pkl")

# load training data & best parameters

logging.info("Loading X_train_scaled & y_train from '%s'", INPUT_DATA_FOLDER)
X_train_scaled = np.load(INPUT_DATA_FOLDER / "X_train_scaled.npy")
y_train = np.load(INPUT_DATA_FOLDER / "y_train.npy")

with open(INPUT_PARAMETERS_FILE, 'rb') as f:
    best_params = pickle.load(f)

logging.info("Loaded from '%s' the parameters: %s", INPUT_PARAMETERS_FILE, best_params)

# Model training

logging.info("Training Random Forest Regressor with best parameters")

rf_reg = RandomForestRegressor(**best_params, random_state=SEED)

rf_reg.fit(X_train_scaled, y_train)

# Saving model

logging.info("Saving regressor to '%s'", OUTPUT_MODEL_FILE)
joblib.dump(rf_reg, OUTPUT_MODEL_FILE);