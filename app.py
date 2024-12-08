import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

@app.route('/predict', methods=['POST','GET'])
def predict():
    json_ = request.json
    print(json_)

    query = pd.DataFrame([json_])

    query = pd.get_dummies(query)
    query = query.reindex(columns=model_columns, fill_value=0)

    # Scale numerical features
    numerical_columns = ['OCC_HOUR', 'REPORT_HOUR', 'BIKE_COST',
                         'LONG_WGS84', 'LAT_WGS84', 'x', 'y', 'BIKE_SPEED']
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler()
    query[numerical_columns] = scaler.fit_transform(query[numerical_columns])

    # Make prediction
    prediction = lr.predict(query)
    return jsonify({'prediction': str(prediction)})

if __name__ == '__main__':
    lr = joblib.load('random_forest_model.pkl')  # Load the model
    model_columns = joblib.load('model_columns.pkl')  # Load model columns

    app.run(debug=True)