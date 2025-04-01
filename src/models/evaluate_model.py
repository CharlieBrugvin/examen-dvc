"""Evaluation du modèle. 

Finalement, en utilisant le modèle entraîné on évaluera ses performances et on fera des prédictions avec ce modèle de sorte qu'à la fin de ce script on aura un nouveau dataset dans data qui contiendra les predictions ainsi qu'un fichier scores.json dans le dossier metrics qui récupérera les métriques d'évaluation de notre modèle (i.e. mse, r2, etc).
"""

from pathlib import Path
import logging

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

logging.basicConfig(level=logging.INFO)

INPUT_DATA_FOLDER = Path("data/processed_data")
INPUT_MODEL_FILE = Path("models/regressor_model.pkl")

OUTPUT_DATA_FOLDER = Path("data/processed_data")
OUTPUT_METRIC_FILE = Path("metrics/scores.json")

# Loading model & test data

logging.info("Loading X_test_scaled & y_test from '%s'", INPUT_DATA_FOLDER)

X_test_scaled = np.load(INPUT_DATA_FOLDER / "X_test_scaled.npy")
y_test = np.load(INPUT_DATA_FOLDER / "y_test.npy")

logging.info("Loading regressor from '%s'", INPUT_MODEL_FILE)
rf_reg = joblib.load(INPUT_MODEL_FILE)

# predicting test data

logging.info("Predicting on X_test_scaled")
y_test_pred = rf_reg.predict(X_test_scaled)

# calculating various metrics

logging.info("Calculating various metrics")
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

logging.info(f"MAE={mae:.3f}, MSE={mse:.3f}, R²={r2:.3f}")

# saving predictions and metrics

pred_filepath = OUTPUT_DATA_FOLDER / "y_test_pred.npy"
logging.info("Saving predictions to '%s'", pred_filepath)
np.save(pred_filepath, y_test_pred)

logging.info("Saving metrics to '%s'", OUTPUT_METRIC_FILE)
OUTPUT_METRIC_FILE.write_text(
            json.dumps(
                {"mae": mae, "mse": mse, "r2": r2}, 
                indent=4
            )
        )