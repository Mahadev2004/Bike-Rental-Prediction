# Bike Rental Demand Predictor

A complete machine learning project for predicting bike rental demand using Python, HTML, CSS, and JavaScript.

## Features

- **Data Processing**: Automated dataset generation and preprocessing
- **Machine Learning**: Multiple regression models (Linear, Random Forest, Gradient Boosting)
- **Interactive UI**: Clean, responsive web interface
- **Real-time Predictions**: Make predictions with custom parameters
- **Visualizations**: Charts and graphs for model comparison and analysis

## Installation

1. Clone the repository:
2. Install Python dependencies:
3. Run the application:

4. Open your browser and navigate to `http://localhost:5000`

## Usage

1. **Data Preprocessing**: Click "Load & Process Data" to generate and prepare the dataset
2. **Model Training**: Train multiple machine learning models with one click
3. **Evaluation**: Compare model performance with interactive charts
4. **Prediction**: Make real-time predictions using trained models

## Project Structure


## Technologies Used

- **Backend**: Python, Flask, scikit-learn, pandas, numpy
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Charts**: Chart.js
- **Icons**: FontAwesome

## Model Features

The system uses the following features for prediction:
- Season (Spring, Summer, Fall, Winter)
- Weather conditions (Clear, Mist, Light Rain, Heavy Rain)
- Temperature, Humidity, Wind Speed
- Weekend/Holiday indicators
- Temporal features (cyclical encoding)
- Interaction features

## API Endpoints

- `POST /api/preprocess` - Process and prepare dataset
- `POST /api/train` - Train machine learning models
- `GET /api/feature_importance` - Get feature importance scores
- `POST /api/predict` - Make predictions
- `GET /api/comparison` - Get model comparison data

## License

MIT License
