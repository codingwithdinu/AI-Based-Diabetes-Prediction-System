import json
import math
import pickle
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "diabetes_model.pkl"
SCALER_PATH = ROOT / "scaler.pkl"

_model = None
_scaler = None


def load_artifacts():
    global _model, _scaler
    if _model is None or _scaler is None:
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            raise FileNotFoundError("Model artifacts are missing from the deployment.")
        with MODEL_PATH.open("rb") as model_file:
            _model = pickle.load(model_file)
        with SCALER_PATH.open("rb") as scaler_file:
            _scaler = pickle.load(scaler_file)
    return _model, _scaler


class handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/api/predict", ""):
            self._send(200, {"status": "ok", "message": "GlucoWise prediction API is ready."})
        else:
            self._send(404, {"error": "Not found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 16_384:
                return self._send(400, {"error": "Invalid request body."})

            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            values = payload.get("features") if isinstance(payload, dict) else None
            if not isinstance(values, list) or len(values) != 8:
                return self._send(400, {"error": "Exactly eight numeric health measurements are required."})

            try:
                numbers = [float(value) for value in values]
            except (TypeError, ValueError):
                return self._send(400, {"error": "All health measurements must be numeric."})

            if not all(math.isfinite(value) for value in numbers):
                return self._send(400, {"error": "Measurements must be finite numbers."})

            pregnancies, glucose, pressure, skin, insulin, bmi, pedigree, age = numbers
            if not (0 <= pregnancies <= 20 and 0 < glucose <= 300 and
                    0 < pressure <= 200 and 0 <= skin <= 100 and
                    0 <= insulin <= 1000 and 0 < bmi <= 100 and
                    0 <= pedigree <= 3 and 1 <= age <= 120):
                return self._send(400, {"error": "One or more measurements are outside the accepted input range."})

            model, scaler = load_artifacts()
            matrix = np.asarray([numbers], dtype=float)
            transformed = scaler.transform(matrix)
            prediction = int(model.predict(transformed)[0])

            if prediction not in (0, 1):
                return self._send(500, {"error": "The model returned an unexpected result."})

            return self._send(200, {
                "prediction": prediction,
                "label": "Higher-risk flag" if prediction == 1 else "Not flagged by this model",
                "disclaimer": "Educational model output only; not a medical diagnosis."
            })
        except json.JSONDecodeError:
            return self._send(400, {"error": "Request must contain valid JSON."})
        except FileNotFoundError as exc:
            return self._send(500, {"error": str(exc)})
        except Exception:
            return self._send(500, {"error": "Prediction service could not complete the request."})

    def log_message(self, format, *args):
        return
