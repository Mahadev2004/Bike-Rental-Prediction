from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
from datetime import datetime, timedelta
import os

app = Flask(__name__, static_folder='static', static_url_path='')

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        return super(NumpyEncoder, self).default(obj)

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif hasattr(obj, 'item'):
        return obj.item()
    return obj

class BikeRentalPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = []
        self.processed_data = None

    def generate_realistic_data(self, num_days=730):  # 2 years of data
        """Generate realistic bike rental dataset"""
        np.random.seed(42)
        data = []
        start_date = datetime(2022, 1, 1)
        
        for i in range(num_days):
            current_date = start_date + timedelta(days=i)
            
            # Date features
            day_of_week = current_date.weekday()
            month = current_date.month
            day = current_date.day
            is_weekend = 1 if day_of_week >= 5 else 0
            is_holiday = 1 if np.random.random() < 0.08 else 0
            
            # Season (1: Spring, 2: Summer, 3: Fall, 4: Winter)
            season = ((month - 1) // 3) + 1
            
            # Weather simulation with realistic patterns
            base_temp = 15 + 15 * np.sin((month - 1) * np.pi / 6)  # Seasonal temperature
            temperature = np.clip(base_temp + np.random.normal(0, 8), -10, 40)
            
            # Humidity varies with season and weather
            base_humidity = 60 + 10 * np.sin((month - 1) * np.pi / 6 + np.pi)
            humidity = np.clip(base_humidity + np.random.normal(0, 15), 20, 95)
            
            # Wind speed
            wind_speed = np.clip(np.random.exponential(8), 0, 40)
            
            # Weather situation with seasonal bias
            weather_prob = np.random.random()
            seasonal_rain_factor = 0.3 if season in [1, 4] else 0.15  # More rain in spring/winter
            
            if weather_prob < 0.6 - seasonal_rain_factor:
                weather_sit = 1  # Clear
            elif weather_prob < 0.8:
                weather_sit = 2  # Mist
            elif weather_prob < 0.95:
                weather_sit = 3  # Light Rain
            else:
                weather_sit = 4  # Heavy Rain
            
            # Calculate rental count with complex interactions
            base_count = 150
            
            # Temperature effect (optimal around 20-25°C)
            if 20 <= temperature <= 25:
                temp_factor = 1.4
            elif 15 <= temperature <= 30:
                temp_factor = 1.2
            elif 10 <= temperature <= 35:
                temp_factor = 1.0
            else:
                temp_factor = 0.6
            
            # Weather effect
            weather_factors = {1: 1.3, 2: 1.0, 3: 0.6, 4: 0.3}
            weather_factor = weather_factors[weather_sit]
            
            # Day and time effects
            if is_weekend:
                day_factor = 1.4  # More rentals on weekends
            else:
                day_factor = 1.0
            
            holiday_factor = 0.7 if is_holiday else 1.0
            
            # Seasonal effects
            seasonal_factors = {1: 1.0, 2: 1.5, 3: 1.2, 4: 0.6}
            seasonal_factor = seasonal_factors[season]
            
            # Hour simulation (simplified daily pattern)
            hour_factor = 1.0 + 0.3 * np.sin((i % 7) * np.pi / 3)  # Weekly pattern
            
            # Calculate final count
            count = (base_count * temp_factor * weather_factor * 
                    day_factor * holiday_factor * seasonal_factor * hour_factor)
            
            # Add noise and ensure minimum
            count = max(5, int(count + np.random.normal(0, 25)))
            
            data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'season': int(season),
                'month': int(month),
                'day': int(day),
                'weekday': int(day_of_week),
                'is_weekend': int(is_weekend),
                'is_holiday': int(is_holiday),
                'weather_situation': int(weather_sit),
                'temperature': float(round(temperature, 2)),
                'humidity': float(round(humidity, 2)),
                'wind_speed': float(round(wind_speed, 2)),
                'count': int(count)
            })
        
        return pd.DataFrame(data)

    def preprocess_data(self):
        """Preprocess the data for machine learning"""
        if self.data is None:
            self.data = self.generate_realistic_data()
        
        # Feature engineering
        df = self.data.copy()
        
        # Create cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        
        # Temperature categories
        df['temp_category'] = pd.cut(df['temperature'], 
                                   bins=[-np.inf, 0, 10, 20, 30, np.inf], 
                                   labels=[0, 1, 2, 3, 4]).astype(int)
        
        # Interaction features
        df['temp_humidity'] = df['temperature'] * df['humidity'] / 100
        df['season_weather'] = df['season'] * df['weather_situation']
        
        # Select features
        feature_columns = [
            'season', 'is_weekend', 'is_holiday', 'weather_situation',
            'temperature', 'humidity', 'wind_speed',
            'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos',
            'temp_category', 'temp_humidity', 'season_weather'
        ]
        
        X = df[feature_columns]
        y = df['count']
        
        self.feature_names = feature_columns
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Scale features for linear regression
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Convert sample data to Python native types
        sample_data = df.head(20).to_dict('records')
        sample_data_clean = []
        for record in sample_data:
            clean_record = {}
            for key, value in record.items():
                if isinstance(value, (np.integer, np.int64, np.int32)):
                    clean_record[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    clean_record[key] = float(value)
                else:
                    clean_record[key] = value
            sample_data_clean.append(clean_record)
        
        # Store processed data info with proper type conversion
        self.processed_data = {
            'total_samples': int(len(df)),
            'train_samples': int(len(self.X_train)),
            'test_samples': int(len(self.X_test)),
            'features': int(len(feature_columns)),
            'feature_names': feature_columns,
            'sample_data': sample_data_clean,
            'data_stats': {
                'mean_rentals': float(df['count'].mean()),
                'max_rentals': int(df['count'].max()),
                'min_rentals': int(df['count'].min()),
                'std_rentals': float(df['count'].std())
            },
            'weather_distribution': {str(k): int(v) for k, v in df['weather_situation'].value_counts().to_dict().items()},
            'seasonal_distribution': {str(k): int(v) for k, v in df['season'].value_counts().to_dict().items()}
        }
        
        return self.processed_data

    def train_models(self):
        """Train multiple regression models"""
        if self.X_train is None:
            raise ValueError("Data must be preprocessed first")
        
        # Define models with better parameters
        models_config = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=150, 
                max_depth=8,
                learning_rate=0.1,
                min_samples_split=5,
                random_state=42
            )
        }
        
        results = {}
        
        for name, model in models_config.items():
            print(f"Training {name}...")
            
            # Train model
            if name == 'Linear Regression':
                model.fit(self.X_train_scaled, self.y_train)
                train_pred = model.predict(self.X_train_scaled)
                test_pred = model.predict(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                train_pred = model.predict(self.X_train)
                test_pred = model.predict(self.X_test)
            
            # Calculate comprehensive metrics
            train_mae = mean_absolute_error(self.y_train, train_pred)
            test_mae = mean_absolute_error(self.y_test, test_pred)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
            train_r2 = r2_score(self.y_train, train_pred)
            test_r2 = r2_score(self.y_test, test_pred)
            
            # Calculate MAPE
            test_mape = np.mean(np.abs((self.y_test - test_pred) / self.y_test)) * 100
            
            # Store model and results
            self.models[name] = model
            
            # Convert all results to Python native types
            results[name] = {
                'train_mae': float(round(train_mae, 2)),
                'test_mae': float(round(test_mae, 2)),
                'train_rmse': float(round(train_rmse, 2)),
                'test_rmse': float(round(test_rmse, 2)),
                'train_r2': float(round(train_r2, 4)),
                'test_r2': float(round(test_r2, 4)),
                'test_mape': float(round(test_mape, 2)),
                'predictions': [float(x) for x in test_pred.tolist()[:100]],
                'actual': [float(x) for x in self.y_test.tolist()[:100]],
                'residuals': [float(x) for x in (self.y_test - test_pred).tolist()[:100]]
            }
        
        return results

    def get_feature_importance(self, model_name='Random Forest'):
        """Get feature importance for tree-based models"""
        if model_name not in self.models:
            return {}
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importance))
            # Convert to Python native types and sort by importance
            feature_importance = {k: float(v) for k, v in feature_importance.items()}
            sorted_features = dict(sorted(feature_importance.items(), 
                                        key=lambda x: x[1], reverse=True))
            return sorted_features
        
        return {}

    def predict_single(self, features, model_name='Random Forest'):
        """Make a single prediction"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        features_array = np.array([features])
        
        if model_name == 'Linear Regression':
            features_array = self.scaler.transform(features_array)
        
        prediction = model.predict(features_array)[0]
        return float(max(0, round(prediction, 0)))

    def get_model_comparison(self):
        """Get comparison data for all trained models"""
        if not self.models:
            return {}
        
        comparison = {}
        for name, model in self.models.items():
            if name == 'Linear Regression':
                predictions = model.predict(self.X_test_scaled)
            else:
                predictions = model.predict(self.X_test)
            
            mae = mean_absolute_error(self.y_test, predictions)
            rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
            r2 = r2_score(self.y_test, predictions)
            
            comparison[name] = {
                'mae': float(round(mae, 2)),
                'rmse': float(round(rmse, 2)),
                'r2': float(round(r2, 4))
            }
        
        return comparison

# Initialize predictor
predictor = BikeRentalPredictor()

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/preprocess', methods=['POST'])
def preprocess_data():
    try:
        result = predictor.preprocess_data()
        # Additional safety conversion
        clean_result = convert_numpy_types(result)
        return jsonify({'success': True, 'data': clean_result})
    except Exception as e:
        print(f"Preprocessing error: {str(e)}")  # Debug print
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/train', methods=['POST'])
def train_models():
    try:
        results = predictor.train_models()
        # Additional safety conversion
        clean_results = convert_numpy_types(results)
        return jsonify({'success': True, 'results': clean_results})
    except Exception as e:
        print(f"Training error: {str(e)}")  # Debug print
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/feature_importance', methods=['GET'])
def get_feature_importance():
    try:
        model_name = request.args.get('model', 'Random Forest')
        importance = predictor.get_feature_importance(model_name)
        clean_importance = convert_numpy_types(importance)
        return jsonify({'success': True, 'importance': clean_importance})
    except Exception as e:
        print(f"Feature importance error: {str(e)}")  # Debug print
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = data['features']
        model_name = data.get('model', 'Random Forest')
        prediction = predictor.predict_single(features, model_name)
        return jsonify({'success': True, 'prediction': float(prediction)})
    except Exception as e:
        print(f"Prediction error: {str(e)}")  # Debug print
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/comparison', methods=['GET'])
def get_comparison():
    try:
        comparison = predictor.get_model_comparison()
        clean_comparison = convert_numpy_types(comparison)
        return jsonify({'success': True, 'comparison': clean_comparison})
    except Exception as e:
        print(f"Comparison error: {str(e)}")  # Debug print
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/dataset', methods=['GET'])
def get_dataset():
    try:
        if predictor.processed_data is None:
            predictor.preprocess_data()
        
        clean_data = convert_numpy_types(predictor.processed_data['sample_data'])
        return jsonify({'success': True, 'data': clean_data})
    except Exception as e:
        print(f"Dataset error: {str(e)}")  # Debug print
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
