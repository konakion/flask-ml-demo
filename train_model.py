# train_model.py
"""
Dieses Script lädt den Iris-Datensatz, trainiert ein einfaches ML-Modell
(Logistische Regression) für **alle drei Klassen** und speichert das Modell
als model.pkl ab.

Schritte:
1. Iris-Daten laden
2. Trainings-/Testdaten erzeugen
3. Modell trainieren (multiklassig)
4. Genauigkeit ausgeben
5. Modell + Feature-Namen speichern
"""

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Datensatz laden
# -----------------------------
# Iris-Datensatz hat:
# - 150 Samples
# - 4 Features (Sepal/Petal Länge/Breite)
# - 3 Klassen (0, 1, 2)
iris = load_iris()
X = iris.data        # Feature-Matrix (shape: [n_samples, 4])
y = iris.target      # Zielvariable (0, 1, 2)

# KEIN Filter => Wir verwenden alle drei Klassen.

# -----------------------------
# 2. Trainings-/Testdaten erzeugen
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,      # 20% als Testdaten
    random_state=42,    # Reproduzierbarkeit
    stratify=y          # sorgt für ähnliche Klassenverteilung in Train/Test
)

# -----------------------------
# 3. Modell definieren & trainieren
# -----------------------------
# LogisticRegression kann auch für mehrere Klassen verwendet werden.
# multi_class="auto": wählt je nach Solver die passende Strategie.
model = LogisticRegression(max_iter=1000)  # max_iter erhöht, falls es sonst nicht konvergiert
model.fit(X_train, y_train)

# -----------------------------
# 4. Performance auf Testdaten
# -----------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Modell-Accuracy (3 Klassen): {acc:.3f}")

# -----------------------------
# 5. Modell + Feature-Namen speichern
# -----------------------------
joblib.dump(model, "model.pkl")
joblib.dump(iris.feature_names, "feature_names.pkl")

print("model.pkl und feature_names.pkl wurden gespeichert (3-Klassen-Modell).")