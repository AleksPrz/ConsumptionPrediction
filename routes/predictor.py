from flask import jsonify, request, Blueprint, make_response
import pandas as pd
import numpy as np
import json
import joblib

predictor_bp = Blueprint("predictor", __name__)

@predictor_bp.route('/')
def hello():
    return 'Hello world'


@predictor_bp.route('/predict', methods = ['POST'])
def predict():
    csv_file = request.files.get('file')
    if csv_file:
        df = pd.read_csv(csv_file)

        with open("mappers/city_mean_sales.json") as f:
            city_means = json.load(f)

        print(df.info())

        df["MEAN_ORIGIN_CONSUPTION"] = df["ORIGEN"].map(city_means["origin"])
        df["MEAN_DEST_CONSUPTION"] = df["DESTINO"].map(city_means["destination"])

        df['LOST SALES'] = 0

        df['FECHA'] = pd.to_datetime(df['FECHA'], format='%d/%m/%y', errors='coerce')
        df['DEPARTUTE LOCAL TIME'] = pd.to_datetime(df['DEPARTUTE LOCAL TIME'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        df['ARRIVAL LOCAL TIME'] = pd.to_datetime(df['ARRIVAL LOCAL TIME'], format='%d/%m/%Y %H:%M:%S', errors='coerce')

        df['MONTH'] = df['FECHA'].dt.month
        df['DAY_OF_WEEK'] = df['FECHA'].dt.weekday 

        # Create cyclic variables
        df['month_sin'] = np.sin(2 * np.pi * df['MONTH']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['MONTH']/12)
        df['dow_sin'] = np.sin(2 * np.pi * df['DAY_OF_WEEK']/7)
        df['dow_cos'] = np.cos(2 * np.pi * df['DAY_OF_WEEK']/7)

        df['DURATION_MIN'] = (df['ARRIVAL LOCAL TIME'] - df['DEPARTUTE LOCAL TIME']).dt.total_seconds() / 60 

        # Extracting hour
        df['HOUR'] = df['DEPARTUTE LOCAL TIME'].dt.hour

        slots = ['EarlyMorning', 'Morning', 'Noon', 'Afternoon', 'Evening', 'LateNight']

        for slot in slots:
            df[f'hour_{slot}'] = 0

        for idx, row in df.iterrows():
            slot = hour_slot(row['HOUR'])
            df.at[idx, f'hour_{slot}'] = 1

        df = category_dummies(df)

        drop_cols = ['FECHA', 'DEPARTUTE LOCAL TIME', 'ARRIVAL LOCAL TIME', 'MONTH', 'DAY_OF_WEEK', 'HOUR', 'ORIGEN', 'DESTINO', 'CATEGORY', 'SUPERCATEGORY', 'ITEM CODE', 'unit_price', 'item_name', 'quantity']
        df = df.drop(columns=drop_cols)

        # Models
        model = joblib.load('models/model.pkl')
        scaler = joblib.load('models/scaler.pkl')

        print(df.info())

        numerical_columns = ['PASSENGERS','DURATION_MIN','MEAN_ORIGIN_CONSUPTION','MEAN_DEST_CONSUPTION', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'LOST SALES']
        X_scaled = df.copy()
        X_scaled[numerical_columns] = scaler.transform(df[numerical_columns])
        predictions = model.predict(X_scaled)
        predictions = np.round(predictions).astype(int)
        predictions = predictions.tolist()

        return jsonify(predictions)

        
def hour_slot(hour):
            if 0 <= hour <= 5:
                return 'EarlyMorning'
            elif 6 <= hour <= 11:
                return 'Morning'
            elif 12 <= hour <= 13:
                return 'Noon'
            elif 14 <= hour <= 17:
                return 'Afternoon'
            elif 18 <= hour <= 21:
                return 'Evening'
            else:
                return 'LateNight'
            

def category_dummies(df):
    categories = [
        'Accessories', 'Alcohol', 'Cold Drink', 'Confectionery', 'Fresh Food',
        'Gents Fragrance', 'Hot Drink', 'Hot Food', 'Ladies Fragrance',
        'Logo', 'Savoury Snacks', 'Skincare & Make-up', 'Sweet Snacks',
        'Tobacco', 'Tobacco.'
    ]

    supercategories = ['BISTRO', 'BOUTIQUE', 'DUTY FREE']

    for cat in categories:
        df[f'category_{cat}'] = 0
    
    for idx, row in df.iterrows():
        cat_value = row['CATEGORY']
        
        if cat_value == 'Accessories':
            df.at[idx, 'category_Accessories'] = 1
        elif cat_value == 'Alcohol':
            df.at[idx, 'category_Alcohol'] = 1
        elif cat_value == 'Cold Drink':
            df.at[idx, 'category_Cold Drink'] = 1
        elif cat_value == 'Confectionery':
            df.at[idx, 'category_Confectionery'] = 1
        elif cat_value == 'Fresh Food':
            df.at[idx, 'category_Fresh Food'] = 1
        elif cat_value == 'Gents Fragrance':
            df.at[idx, 'category_Gents Fragrance'] = 1
        elif cat_value == 'Hot Drink':
            df.at[idx, 'category_Hot Drink'] = 1
        elif cat_value == 'Hot Food':
            df.at[idx, 'category_Hot Food'] = 1
        elif cat_value == 'Ladies Fragrance':
            df.at[idx, 'category_Ladies Fragrance'] = 1
        elif cat_value == 'Logo':
            df.at[idx, 'category_Logo'] = 1
        elif cat_value == 'Savoury Snacks':
            df.at[idx, 'category_Savoury Snacks'] = 1
        elif cat_value == 'Skincare & Make-up':
            df.at[idx, 'category_Skincare & Make-up'] = 1
        elif cat_value == 'Sweet Snacks':
            df.at[idx, 'category_Sweet Snacks'] = 1
        elif cat_value == 'Tobacco':
            df.at[idx, 'category_Tobacco'] = 1
        elif cat_value == 'Tobacco.':
            df.at[idx, 'category_Tobacco.'] = 1

        for sup in supercategories:
            df[f'supercategory_{sup}'] = 0

        for idx, row in df.iterrows():
            sup_value = row['SUPERCATEGORY']
            
            if sup_value == 'BISTRO':
                df.at[idx, 'supercategory_BISTRO'] = 1
            elif sup_value == 'BOUTIQUE':
                df.at[idx, 'supercategory_BOUTIQUE'] = 1
            elif sup_value == 'DUTY FREE':
                df.at[idx, 'supercategory_DUTY FREE'] = 1
    return df