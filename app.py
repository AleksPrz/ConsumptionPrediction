from flask import Flask
from flask_cors import CORS
from routes import predictor_bp

app = Flask(__name__)
app.register_blueprint(predictor_bp)

CORS(app, supports_credentials=True)

if __name__ == '__main__':
    app.run(debug=True, port=4000)