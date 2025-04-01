"""Split des données en ensemble d'entraînement et de test.

Notre variable cible est silica_concentrate et se trouve dans la dernière colonne du dataset.
L'issu de ce script seront 4 datasets (X_test, X_train, y_test, y_train) que vous pouvez stocker dans data/processed.
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)


INPUT_DATA_PATH = Path('data/raw_data/raw.csv')
SEED = 42
OUTPUT_FOLDER = Path("data/processed_data")

# loading data & getting X & y datasets

logging.info("Reading data from '%s'", INPUT_DATA_PATH)
df = pd.read_csv(INPUT_DATA_PATH)

logging.info("Getting X & y datasets")
X = df[[
    'ave_flot_air_flow', 
    'ave_flot_level', 
    'iron_feed',
    'starch_flow',
    'amina_flow',
    'ore_pulp_flow',
    'ore_pulp_pH',
    'ore_pulp_density',
]].values

y = df['silica_concentrate'].values

# Train / test split

logging.info("Splitting into a train & test set")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)

# Saving

logging.info("Saving the 4 subsets into : '%s'", OUTPUT_FOLDER)
np.save(OUTPUT_FOLDER / "X_train", X_train)
np.save(OUTPUT_FOLDER / "X_test", X_test)
np.save(OUTPUT_FOLDER / "y_train", y_train)
np.save(OUTPUT_FOLDER / "y_test", y_test)