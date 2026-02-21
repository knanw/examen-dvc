import joblib
import pandas as pd 
import numpy as np
from joblib import load
import json
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

X_train = pd.read_csv('data/preprocessed/X_train.csv')
X_test = pd.read_csv('data/preprocessed/X_test.csv')
y_train = pd.read_csv('data/preprocessed/y_train.csv')
y_test = pd.read_csv('data/preprocessed/y_test.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

def main(repo_path):
    model = load(repo_path / "models/trained_model.joblib")
    scaler = joblib.load('models/scaler.pkl')

    X_test_scaled = scaler.transform(X_test)

    predictions = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    scores = {"mae": mae, "r2": r2}
    accuracy_path = repo_path / "metrics/scores.json"
    accuracy_path.write_text(json.dumps(scores))

    pd.DataFrame(predictions, columns=['predicted_silica']).to_csv(repo_path / 'data/predictions.csv', index=False)

if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent
    main(repo_path)