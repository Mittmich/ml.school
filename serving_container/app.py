# | filename: app.py
# | code-line-numbers: true

import tarfile
import pandas as pd
import io
import tempfile
import numpy as np
import joblib

from flask import Flask, request, jsonify
from pathlib import Path
from tensorflow import keras


MODEL_PATH = Path(__file__).parent


class Model:
    model = None
    target_transformer = None
    input_transformer = None

    def load(self):
        """
        Extracts the model package and loads the model in memory
        if it hasn't been loaded yet.
        """
        # We want to load the model only if it is not loaded yet.
        if not Model.model:
            # Before we load the model, we need to extract it in
            # a temporal directory.

            with tempfile.TemporaryDirectory() as directory:
                with tarfile.open(MODEL_PATH / "model.tar.gz") as tar:
                    tar.extractall(path=directory)

                Model.model = keras.models.load_model(Path(directory) / "001")
                # load the sklearn pipelines usin gjoblib
                Model.target_transformer = joblib.load(Path(directory) / "target.joblib")
                Model.input_transformer = joblib.load(Path(directory) / "features.joblib")

    def check_data(self, data):
        """
        Check whether the data is in the expected format.
        """
        self.load()
        try:
            self.input_transformer.transform(self.parse_data(data))
        except Exception as e:
            return False
        return True

    def parse_data(self, data):
        """
        Parses the data received in the request.
        """
        # We expect the data to be a CSV string.
        self.load()
        return pd.read_csv(io.StringIO(data), names=Model.input_transformer.feature_names_in_, header=None)

    def predict(self, data):
        """
        Generates predictions for the supplied data.
        """
        self.load()
        # prepare data
        data = self.parse_data(data)
        # We need to transform the data before making a prediction.
        data = Model.input_transformer.transform(data)
        predictions =  Model.model.predict(data)
        # get index with highest probability
        prediction = np.argmax(predictions[0], axis=-1)
        species = Model.target_transformer.named_transformers_["species"].categories_[0][prediction]
        return species, float(predictions[0][prediction])


app = Flask(__name__)
model = Model()


@app.route("/refresh/", methods=["POST"])
def reload_model():
    """
    Reloads the model in memory.
    """
    model.model = None
    model.load()
    return "Model reloaded!"

@app.route("/predict/", methods=["POST"])
def predict():
    data = request.data.decode("utf-8")
    # check whether input conforms to expected format
    if not model.check_data(data):
        return jsonify({"error": "Invalid input data."}), 400

    prediction, confidence = model.predict(data=data)

    # check accept header
    if "application/csv" in request.headers.get("Accept", ""):
        return f"prediction,confidence\n{prediction},{confidence}"
    return jsonify({"prediction": prediction, "confidence": confidence})
