"""Normalisation des données. 

Comme vous pouvez le noter, les données sont dans des échelles très variés donc une normalisation est nécessaire.
Vous pouvez utiliser des fonctions pré-existantes pour la construction de ce script. 
En sortie, ce script créera deux nouveaux datasets : (X_train_scaled, X_test_scaled) que vous sauvegarderez également dans data/processed.
"""

from pathlib import Path
import logging

import numpy as np
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)

INPUT_DATA_FOLDER = Path("data/processed_data")
OUTPUT_DATA_FOLDER = Path("data/processed_data")

# Loading data

logging.info("Loading X_train & X_test from : '%s'", INPUT_DATA_FOLDER)

X_train = np.load(INPUT_DATA_FOLDER / "X_train.npy")
X_test = np.load(INPUT_DATA_FOLDER / "X_test.npy")

logging.info("Shapes: X_train=%s X_test=%s", 
             X_train.shape, X_test.shape)

# Normalization

logging.info("Fitting Standard Scaler & transforming X_train & X_test ")

X_scaler = StandardScaler().fit(np.vstack([X_train, X_test]))

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# Saving data

logging.info("Saving X_train_scaled & X_test_scaled into : '%s'", OUTPUT_DATA_FOLDER)
np.save(OUTPUT_DATA_FOLDER / "X_train_scaled.npy", X_train_scaled)
np.save(OUTPUT_DATA_FOLDER / "X_test_scaled.npy", X_test_scaled)