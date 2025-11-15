# app.py
"""
Flask-App, die ein zuvor trainiertes ML-Modell (Iris-Datensatz, 3 Klassen)
als Webservice zur Verfügung stellt.

Funktionen:
- GET  /        → einfache HTML-Seite mit Formular und JavaScript
- POST /predict → nimmt Features als JSON entgegen und gibt die Vorhersage zurück
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np

# ---------------------------------------
# 1. Flask-App erstellen
# ---------------------------------------
app = Flask(__name__)

# ---------------------------------------
# 2. Modell und Meta-Daten laden
# ---------------------------------------
# Das Modell wurde in train_model.py mit joblib.dump gespeichert.
model = joblib.load("model.pkl")

# Optional: Feature-Namen laden, falls vorhanden
try:
    feature_names = joblib.load("feature_names.pkl")
except Exception:
    feature_names = None

# Für die Anzeige sind Namen der Klassen praktisch.
# Beim Iris-Datensatz haben wir:
#   0 → setosa
#   1 → versicolor
#   2 → virginica
CLASS_LABELS = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}


# ---------------------------------------
# 3. Startseite: einfache HTML-Oberfläche
# ---------------------------------------
@app.route("/")
def index():
    """
    Gibt eine kleine HTML-Seite zurück, mit der man
    direkt im Browser Vorhersagen testen kann.
    """
    return """
    <!doctype html>
    <html lang="de">
    <head>
      <meta charset="utf-8">
      <title>ML Demo – Iris Prediction</title>
      <style>
        body { font-family: sans-serif; max-width: 600px; margin: 40px auto; }
        label { display: block; margin-top: 10px; }
        input { width: 100%; padding: 6px; }
        button { margin-top: 15px; padding: 8px 12px; }
        #result { margin-top: 20px; font-weight: bold; white-space: pre-line; }
      </style>
    </head>
    <body>
      <h1>ML Demo – Iris Prediction</h1>
      <p>Gib die vier Iris-Features ein und klicke auf <strong>Predict</strong>.</p>

      <label>Sepal length (cm):
        <input id="f1" type="number" step="0.01" value="5.1">
      </label>
      <label>Sepal width (cm):
        <input id="f2" type="number" step="0.01" value="3.5">
      </label>
      <label>Petal length (cm):
        <input id="f3" type="number" step="0.01" value="1.4">
      </label>
      <label>Petal width (cm):
        <input id="f4" type="number" step="0.01" value="0.2">
      </label>

      <button onclick="sendPredict()">Predict</button>

      <div id="result"></div>

      <script>
        // Diese Funktion wird beim Klick auf den Button aufgerufen.
        async function sendPredict() {
          const f1 = parseFloat(document.getElementById('f1').value);
          const f2 = parseFloat(document.getElementById('f2').value);
          const f3 = parseFloat(document.getElementById('f3').value);
          const f4 = parseFloat(document.getElementById('f4').value);

          const payload = {
            features: [f1, f2, f3, f4]
          };

          const resDiv = document.getElementById('result');
          resDiv.textContent = "Bitte warten...";

          try {
            const response = await fetch('/predict', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload)
            });

            if (!response.ok) {
              const errText = await response.text();
              resDiv.textContent = "Fehler: " + errText;
              return;
            }

            const data = await response.json();

            // Schön formatiertes Ergebnis
            const cls = data.class_label;
            const pred = data.prediction;
            const probs = data.probabilities;

            let probText = "";
            if (probs && probs.length === 3) {
              probText =
                "Wahrscheinlichkeiten:\\n" +
                "  setosa:     " + probs[0].toFixed(3) + "\\n" +
                "  versicolor: " + probs[1].toFixed(3) + "\\n" +
                "  virginica:  " + probs[2].toFixed(3);
            }

            resDiv.textContent =
              "Vorhergesagte Klasse: " + cls + " (Index: " + pred + ")\\n" +
              probText;

          } catch (err) {
            resDiv.textContent = "Fehler beim Request: " + err;
          }
        }
      </script>
    </body>
    </html>
    """


# ---------------------------------------
# 4. API-Endpunkt: /predict
# ---------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Erwartet JSON der Form:
        { "features": [f1, f2, f3, f4] }

    Gibt JSON zurück mit:
        - prediction: Klassenindex (0, 1, 2)
        - class_label: Klartext-Label ("setosa", ...)
        - probabilities: Liste der Klass-Wahrscheinlichkeiten
        - feature_names (optional)
    """
    data = request.get_json()

    # Minimaler Check: Ist überhaupt JSON da und enthält es 'features'?
    if not data or "features" not in data:
        return jsonify({"error": "Bitte JSON mit Key 'features' senden"}), 400

    features = data["features"]

    # Sicherstellen, dass genau 4 Features kommen (Iris-Standard)
    if len(features) != 4:
        return jsonify({"error": "Es werden genau 4 Features erwartet."}), 400

    # In numpy-Array und auf 2D-Form bringen: shape (1, 4)
    X = np.array(features).reshape(1, -1)

    # Vorhersage und Wahrscheinlichkeiten vom Modell holen
    pred = int(model.predict(X)[0])      # 0, 1 oder 2
    proba = model.predict_proba(X)[0].tolist()

    # Klassenlabel in Klartext bestimmen
    class_label = CLASS_LABELS.get(pred, f"Unbekannte Klasse {pred}")

    response = {
        "prediction": pred,
        "class_label": class_label,
        "probabilities": proba,
    }

    # Optional auch die Feature-Namen mit zurückgeben
    if feature_names:
        response["feature_names"] = list(feature_names)

    return jsonify(response)


# ---------------------------------------
# 5. Lokaler Start der App
# ---------------------------------------
if __name__ == "__main__":
    # host="0.0.0.0" → erlaubt Zugriff auch von anderen Geräten im Netzwerk
    # debug=True     → automatischer Reload bei Codeänderungen (für Entwicklung)
    app.run(host="0.0.0.0", port=5000, debug=True)