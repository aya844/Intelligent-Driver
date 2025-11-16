import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)

N = 500
data = pd.DataFrame({
    "TTC": np.random.uniform(0.1, 4, N),
    "Danger_externe": np.random.uniform(0, 1, N),
    "DriverState": np.random.uniform(0, 100, N),
    "Vitesse": np.random.uniform(0, 140, N),
    "Freinage": np.random.uniform(0, 1, N),
    "Meteo": np.random.choice([0,1], N)
})

# Label simplifié
data["Crash"] = (
    (data.TTC < 1.0).astype(int) |
    ((data.Danger_externe > 0.7) & (data.DriverState < 40)).astype(int)
)
X = data[["TTC", "Danger_externe", "DriverState", "Vitesse", "Freinage", "Meteo"]]
y = data["Crash"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Sauvegarde modèle
import joblib
joblib.dump(model, "fusion_model.pkl")
