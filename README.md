# ⏱️ Food Delivery Time Prediction - Regression ML

A **machine learning regression system** for predicting food delivery times using time series analysis, gradient boosting, and feature engineering.

## 🎯 Overview

This project includes:
- ✅ Delivery data collection
- ✅ Feature engineering
- ✅ Regression models
- ✅ Time series analysis
- ✅ Real-time predictions
- ✅ Performance metrics
- ✅ Optimization pipeline

## 📦 Data Collection & Preprocessing

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DeliveryDataCollector:
    """Collect delivery data"""
    
    def __init__(self):
        self.data = None
    
    def load_deliveries(self, filepath):
        """Load delivery records"""
        self.data = pd.read_csv(filepath)
        self.data['order_time'] = pd.to_datetime(self.data['order_time'])
        self.data['delivery_time'] = pd.to_datetime(self.data['delivery_time'])
        
        return self.data
    
    def calculate_delivery_duration(self):
        """Calculate time taken"""
        df = self.data.copy()
        
        df['delivery_duration_minutes'] = (
            (df['delivery_time'] - df['order_time']).dt.total_seconds() / 60
        )
        
        return df
    
    def extract_temporal_features(self):
        """Extract time-based features"""
        df = self.data.copy()
        
        # Time of day
        df['order_hour'] = df['order_time'].dt.hour
        df['order_day'] = df['order_time'].dt.dayofweek
        df['order_month'] = df['order_time'].dt.month
        
        # Is peak hours? (lunch/dinner)
        df['is_peak_hours'] = ((df['order_hour'] >= 12) & (df['order_hour'] <= 14)) | \
                              ((df['order_hour'] >= 19) & (df['order_hour'] <= 21))
        
        # Is weekend
        df['is_weekend'] = df['order_day'].isin([5, 6]).astype(int)
        
        # Hour sin/cos for cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['order_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['order_hour'] / 24)
        
        return df
    
    def extract_distance_features(self):
        """Calculate distance metrics"""
        df = self.data.copy()
        
        # Haversine distance
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371  # Earth radius in km
            
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            
            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            
            return R * c
        
        df['distance_km'] = haversine(
            df['restaurant_lat'], df['restaurant_lon'],
            df['delivery_lat'], df['delivery_lon']
        )
        
        # Distance category
        df['distance_category'] = pd.cut(df['distance_km'],
                                        bins=[0, 2, 5, 10, 50],
                                        labels=['nearby', 'close', 'moderate', 'far'])
        
        return df
    
    def extract_restaurant_features(self):
        """Restaurant-based features"""
        df = self.data.copy()
        
        # Restaurant average preparation time
        prep_times = df.groupby('restaurant_id')['preparation_time_minutes'].transform('mean')
        df['restaurant_avg_prep_time'] = prep_times
        
        # Restaurant delivery count (proxy for efficiency)
        delivery_counts = df.groupby('restaurant_id').size().transform('count')
        df['restaurant_delivery_count'] = delivery_counts
        
        # Restaurant rating
        if 'restaurant_rating' in df.columns:
            df['high_rated_restaurant'] = (df['restaurant_rating'] >= 4.5).astype(int)
        
        return df
    
    def create_lag_features(self):
        """Create lag features for time series"""
        df = self.data.copy()
        
        # Sort by time
        df = df.sort_values('order_time')
        
        # Rolling averages for delivery time
        df['delivery_time_ma7'] = df['delivery_duration_minutes'].rolling(7).mean()
        df['delivery_time_ma30'] = df['delivery_duration_minutes'].rolling(30).mean()
        
        # Lag features
        for lag in [1, 7, 24]:
            df[f'delivery_time_lag{lag}h'] = df['delivery_duration_minutes'].shift(lag)
        
        return df
```

## 🧠 Regression Models

```python
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score

class DeliveryTimeRegressor:
    """Predict delivery time"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
    
    def linear_regression(self, X_train, y_train):
        """Baseline regression"""
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        self.models['lr'] = lr
        
        return lr
    
    def ridge_regression(self, X_train, y_train, alpha=1.0):
        """Ridge with L2 regularization"""
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        self.models['ridge'] = ridge
        
        return ridge
    
    def gradient_boosting_regressor(self, X_train, y_train):
        """Gradient Boosting for regression"""
        gbr = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            min_samples_split=5,
            subsample=0.8,
            random_state=42
        )
        
        gbr.fit(X_train, y_train)
        self.models['gbr'] = gbr
        
        return gbr
    
    def random_forest_regressor(self, X_train, y_train):
        """Random Forest regression"""
        rf_reg = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        rf_reg.fit(X_train, y_train)
        self.models['rf_reg'] = rf_reg
        
        return rf_reg
    
    def xgboost_regressor(self, X_train, y_train):
        """XGBoost for fast boosting"""
        import xgboost as xgb
        
        xgb_reg = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            random_state=42,
            tree_method='hist'
        )
        
        xgb_reg.fit(X_train, y_train)
        self.models['xgb'] = xgb_reg
        
        return xgb_reg
    
    def predict_delivery_time(self, features):
        """Predict time for single order"""
        if self.best_model is None:
            raise ValueError("Model not trained")
        
        predicted_time = self.best_model.predict([features])[0]
        
        return max(0, predicted_time)  # Ensure non-negative
```

## 📊 Evaluation Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class DeliveryMetricsEvaluator:
    """Evaluate regression performance"""
    
    @staticmethod
    def evaluate(y_true, y_pred):
        """Calculate metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }
    
    @staticmethod
    def prediction_accuracy(y_true, y_pred, tolerance_minutes=5):
        """% predictions within tolerance"""
        absolute_errors = np.abs(y_true - y_pred)
        within_tolerance = np.sum(absolute_errors <= tolerance_minutes)
        
        accuracy = (within_tolerance / len(y_true)) * 100
        
        return accuracy
    
    @staticmethod
    def underestimation_rate(y_true, y_pred):
        """% predictions lower than actual"""
        underestimated = np.sum(y_pred < y_true)
        
        rate = (underestimated / len(y_true)) * 100
        
        return rate
```

## ⏳ Real-time Prediction Pipeline

```python
class DeliveryPredictionPipeline:
    """End-to-end prediction system"""
    
    def __init__(self, model, scaler, feature_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
    
    def predict_order_delivery(self, order_details):
        """Predict delivery time for new order"""
        # Extract features
        features = self.extract_order_features(order_details)
        
        # Scale
        features_scaled = self.scaler.transform([features])
        
        # Predict
        predicted_minutes = self.model.predict(features_scaled)[0]
        predicted_minutes = max(10, predicted_minutes)  # Minimum 10 min
        
        # Convert to datetime
        estimated_delivery = order_details['order_time'] + timedelta(minutes=predicted_minutes)
        
        return {
            'predicted_minutes': predicted_minutes,
            'estimated_delivery_time': estimated_delivery,
            'confidence_interval': self.get_confidence_interval(predicted_minutes)
        }
    
    def extract_order_features(self, order_dict):
        """Extract features from order"""
        features = []
        for fname in self.feature_names:
            if fname in order_dict:
                features.append(order_dict[fname])
        
        return np.array(features)
    
    def get_confidence_interval(self, prediction):
        """Calculate 95% CI"""
        margin = prediction * 0.2  # 20% margin
        
        return {
            'lower': max(0, prediction - margin),
            'upper': prediction + margin
        }
```

## 💡 Interview Talking Points

**Q: Time series challenges?**
```
Answer:
- Stationarity vs trend
- Seasonality patterns (peak hours)
- Lag features important
- ARIMA vs ML approach
- Recursive vs direct prediction
```

**Q: Why underestimate problematic?**
```
Answer:
- Customer satisfaction decrease
- Restaurant reputation risk
- Model credibility loss
- Can overestimate slightly (~10%)
- Cost of wrong forecast
```

## 🌟 Portfolio Value

✅ Regression modeling
✅ Time series analysis
✅ Feature engineering
✅ Multi-model comparison
✅ Real-world application
✅ Performance forecasting
✅ Business impact metrics

---

**Technologies**: Scikit-learn, XGBoost, Pandas

