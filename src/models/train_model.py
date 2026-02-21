import pandas as pd 
import joblib
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

print(joblib.__version__)

X_train = pd.read_csv('data/processed_data/X_train.csv')
X_test = pd.read_csv('data/processed_data/X_test.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Speichern
joblib.dump(scaler, 'models/scaler.pkl')


# GridSearchCV für Hyperparameter-Tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8],
    'min_samples_split': [2, 5],
    'loss': ['squared_error']
}

# TimeSeriesSplit ist wichtig, um die Zeitstruktur nicht zu zerstören
gbr = GradientBoostingRegressor(random_state=42)
tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(gbr, param_grid, cv=tscv, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Beste Parameter speichern
joblib.dump(grid_search.best_params_, 'models/best_params.pkl')
print("Best parameters found: ", grid_search.best_params_)

best_params = joblib.load('models/best_params.pkl')
final_model = GradientBoostingRegressor(**best_params)
final_model.fit(X_train_scaled, y_train)

model_filename = './models/trained_model.joblib'
joblib.dump(final_model, model_filename)
print("Model trained and saved successfully.")
