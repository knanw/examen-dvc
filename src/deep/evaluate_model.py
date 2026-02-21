import pandas as pd 
import numpy as np
from joblib import load
import json
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
from model_definition import SilicaPredictor
import torch

def main(repo_path):
    # 1. Daten laden
    X_test = pd.read_csv(repo_path / 'data/preprocessed/X_test.csv')
    y_test = pd.read_csv(repo_path / 'data/preprocessed/y_test.csv')
    y_test = np.ravel(y_test)

    # 2. Scaler laden (Essentiell für die Rücktransformation!)
    scaler_X = load(repo_path / 'models/scaler_X.pkl')
    scaler_y = load(repo_path / 'models/scaler_y.pkl')

    # 3. Modell laden
    # Hinweis: Input_size muss der Anzahl der Features entsprechen
    input_size = X_test.shape[1] 
    loaded_model = SilicaPredictor(input_size=input_size)
    
    # Nutze .pth für PyTorch Gewichte (nicht .joblib)
    model_path = repo_path / "models/silica_model_pytorch.pth" 
    loaded_model.load_state_dict(torch.load(model_path))
    
    # 4. Evaluieren
    mae, r2, preds = evaluate_pytorch_model(loaded_model, X_test, y_test, scaler_X, scaler_y)

    # 5. Metriken speichern
    metrics = {
        "mae": float(mae), 
        "r2": float(r2)
    }
    accuracy_path = repo_path / "metrics/metrics_deep.json"
    accuracy_path.parent.mkdir(parents=True, exist_ok=True) # Ordner erstellen falls nötig
    
    with open(accuracy_path, 'w') as f:
        json.dump(metrics, f)
        
    print(f"Evaluation fertig. MAE: {mae:.4f}, R2: {r2:.4f}")

def evaluate_pytorch_model(model, X_test, y_test, scaler_X, scaler_y):
    model.eval() # Schaltet Dropout/Batchnorm aus
    
    # Transformation der Features
    X_test_s = scaler_X.transform(X_test)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
    
    with torch.no_grad(): # Deaktiviert die Gradientenberechnung (spart RAM)
        preds_s = model(X_test_t).numpy()
        # WICHTIG: Rücktransformation in den Originalbereich (% Silica)
        preds = scaler_y.inverse_transform(preds_s)
    
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return mae, r2, preds

if __name__ == "__main__":
    # Dynamische Pfadfindung relativ zum Skript-Ort
    current_path = Path(__file__).resolve()
    # Geh so viele Ebenen hoch wie nötig, um zum Repo-Root zu kommen
    repo_root = current_path.parent.parent # Anpassen je nach Ordnerstruktur
    main(repo_root)