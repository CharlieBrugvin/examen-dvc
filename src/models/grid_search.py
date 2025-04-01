"""GridSearch des meilleurs paramètres à utiliser pour la modélisation.

Vous déciderez le modèle de regression à implémenter et des paramètres à tester.
À l'issue de ce script vous aurez les meilleurs paramètres sous forme de fichier .pkl que vous sauvegarderez dans le dossier models.
"""
from pathlib import Path
import logging
import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

logging.basicConfig(level=logging.INFO)


INPUT_DATA_FOLDER = Path("data/processed_data")
# PARAM_GRID = {
#     'n_estimators': [100, 200, 300, 400],
#     'max_depth': [None, 10, 20, 30, 40],
# }
PARAM_GRID = {
    'n_estimators': [300],
}
OUTPUT_PARAM_FILE = Path("models/best_parameters.pkl")

# Loading data 

logging.info("Loading the 4 subsets from : '%s'", INPUT_DATA_FOLDER)

X_train_scaled = np.load(INPUT_DATA_FOLDER / "X_train_scaled.npy")
X_test_scaled = np.load(INPUT_DATA_FOLDER / "X_test_scaled.npy")
y_train = np.load(INPUT_DATA_FOLDER / "y_train.npy")
y_test = np.load(INPUT_DATA_FOLDER / "y_test.npy")

logging.info("Shapes: X_train_scaled=%s X_test_scaled=%s y_train=%s y_test=%s", 
             X_train_scaled.shape, X_test_scaled.shape, y_train.shape, y_test.shape)

# Model initialization and grid search

logging.info("Grid Searching over hyper-parameters")
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=PARAM_GRID,
    scoring='neg_mean_squared_error',
    verbose=2
)

grid_search.fit(X_train_scaled, y_train)

# Saving best parameters

logging.info("Best parameters: %s., Saving them to '%s'", 
             grid_search.best_params_, OUTPUT_PARAM_FILE)
             
with open(OUTPUT_PARAM_FILE, "wb") as f:
    pickle.dump(grid_search.best_params_, f)
