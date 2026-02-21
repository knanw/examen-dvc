import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
from model_definition import SilicaPredictor

X_train = pd.read_csv('data/preprocessed/X_train.csv')
X_test = pd.read_csv('data/preprocessed/X_test.csv')
y_train = pd.read_csv('data/preprocessed/y_train.csv')
y_test = pd.read_csv('data/preprocessed/y_test.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)


# Falls y_train/y_test DataFrames sind:
y_train_values = y_train.values.reshape(-1, 1) if hasattr(y_train, 'values') else y_train.reshape(-1, 1)
y_test_values = y_test.values.reshape(-1, 1) if hasattr(y_test, 'values') else y_test.reshape(-1, 1)


# 1. Daten-Skalierung (Essentiell für Deep Learning!)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_s = scaler_X.fit_transform(X_train)
X_test_s = scaler_X.transform(X_test)

# Target muss als Spalte vorliegen für den Scaler
y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_s = scaler_y.transform(y_test.values.reshape(-1, 1))

# 2. Scaler speichern (sehr wichtig für die spätere Evaluation!)
joblib.dump(scaler_X, 'models/scaler_X.pkl')
joblib.dump(scaler_y, 'models/scaler_y.pkl')

# Konvertierung in PyTorch Tensoren
X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
y_train_t = torch.tensor(y_train_s, dtype=torch.float32)
X_test_t = torch.tensor(X_test_s, dtype=torch.float32)

# 2. Architektur des Neuronalen Netzes
model_dl = SilicaPredictor(X_train_t.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model_dl.parameters(), lr=0.01)

# 3. Training Loop
epochs = 100
for epoch in range(epochs):
    model_dl.train()
    optimizer.zero_grad()
    outputs = model_dl(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 4. Vorhersage
model_dl.eval()
with torch.no_grad():
    raw_preds = model_dl(X_test_t).numpy()
    # Rücktransformation der Skalierung
    predictions_dl = scaler_y.inverse_transform(raw_preds)

print("Deep Learning Vorhersage bereit!")

# Speichern
path = "silica_model_pytorch.pth"
torch.save(model_dl.state_dict(), path)
print(f"Modell gespeichert unter {path}")