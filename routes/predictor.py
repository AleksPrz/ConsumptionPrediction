from flask import jsonify, request, Blueprint, make_response
import pandas as pd

predictor_bp = Blueprint("predictor", __name__)

@predictor_bp.route('/')
def hello():
    return 'Hello world'


@predictor_bp.route('/predict', methods = ['POST'])
def predict():
    csv_file = request.files.get('file')
    if csv_file:
        df = pd.read_csv(csv_file)

    pass