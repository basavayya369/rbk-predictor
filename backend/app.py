from flask import Flask, request, jsonify
import onnxruntime as rt
import numpy as np
import pandas as pd
from flask_cors import CORS

# Path to ONNX model
MODEL_PATH = "trained_pipelineV.onnx"

# Create Flask app
app = Flask(__name__)
CORS(app)

# Load ONNX model
try:
    sess = rt.InferenceSession(MODEL_PATH)
    input_names = [inp.name for inp in sess.get_inputs()]
    output_name = sess.get_outputs()[0].name
    print(f" ONNX model loaded from {MODEL_PATH}")
    print("Expected inputs:", input_names)
except Exception as e:
    print(f" Failed to load ONNX model: {e}")
    sess = None


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Backend running with ONNX",
        "endpoints": ["/predict", "/retrain"]
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Return prediction using ONNX model"""
    if sess is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # Get input JSON
        data = request.get_json()
        print("📥 Incoming data:", data)

        # Convert JSON → numpy with correct shape (n,1)
        input_data = {
            "District": np.array([[data["district"]]], dtype=str),
            "Mandal": np.array([[data["mandal"]]], dtype=str),
            "RBK": np.array([[data["rbk"]]], dtype=str),
            "Season": np.array([[data["season"]]], dtype=str),
            "QTY_MTs": np.array([[float(data["qty"])]], dtype=np.float32),
            "No_Of_Farmers": np.array([[int(data["farmers"])]], dtype=np.float32),
        }

        # Run ONNX inference
        prediction = sess.run([output_name], input_data)[0][0]
        print(" Prediction:", prediction)

        return jsonify({
            "predicted_amount": float(prediction)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/retrain", methods=["POST"])
def retrain():
    """Placeholder retraining endpoint"""
    return jsonify({"message": "Retraining endpoint placeholder"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
