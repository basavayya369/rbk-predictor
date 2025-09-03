from flask import Flask, request, jsonify
import joblib
import shap
import pandas as pd
from flask_cors import CORS

MODEL_PATH = "trained_pipelineV.joblib"

# Create Flask app
app = Flask(__name__)
CORS(app)

# Load the model once
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None


@app.route("/predict", methods=["POST"])
def predict():
    """Return prediction + SHAP values."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # Read request
        data = request.get_json()
        print(" Incoming data:", data)

        # Match training column names
        input_df = pd.DataFrame([{
            "District": data["district"],
            "Mandal": data["mandal"],
            "RBK": data["rbk"],
            "Season": data["season"],
            "QTY_MTs": float(data["qty"]),
            "No_Of_Farmers": int(data["farmers"])
        }])
        print(" Input DataFrame:")
        print(input_df)

        # Prediction
        prediction = model.predict(input_df)[0]
        print(" Prediction:", prediction)

        # SHAP values
        shap_dict = {}
        try:
            transformed = model.named_steps["preprocessor"].transform(input_df)
            explainer = shap.Explainer(model.named_steps["regressor"], transformed)
            shap_values = explainer(transformed)

            feature_names = model.named_steps["preprocessor"].get_feature_names_out()
            shap_dict = {
                feature_names[i]: float(shap_values.values[0][i])
                for i in range(len(feature_names))
            }
            print(" SHAP values generated")
        except Exception as shap_err:
            print("⚠ SHAP calculation failed:", shap_err)

        return jsonify({
            "predicted_amount": float(prediction),
            "shap_values": shap_dict
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/retrain", methods=["POST"])
def retrain():
    """Placeholder retraining endpoint."""
    return jsonify({"message": "Retraining endpoint placeholder"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
