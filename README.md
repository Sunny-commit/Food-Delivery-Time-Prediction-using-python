# 🍕 Food Delivery Time Prediction - Regression Analysis

A **machine learning project predicting food delivery time** based on restaurant features, order characteristics, and delivery partner metrics.

## 🎯 Overview

This project covers:
- ✅ Delivery time prediction
- ✅ Distance and route analysis
- ✅ Time-of-day features
- ✅ Restaurant & order features
- ✅ Traffic pattern modeling
- ✅ Delivery partner performance

## 📊 Delivery Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from haversine import haversine

class FoodDeliveryAnalysis:
    """Analyze delivery data"""
    
    def __init__(self, filepath='food_delivery.csv'):
        self.df = pd.read_csv(filepath)
    
    def explore_data(self):
        """Dataset overview"""
        print(f"Total deliveries: {len(self.df)}")
        print(f"\nDelivery time statistics (minutes):")
        print(self.df['Delivery_Time'].describe())
        
        print(f"\nDistance statistics (km):")
        print(self.df['Distance_km'].describe())
    
    def delivery_performance_metrics(self):
        """Analyze performance"""
        # On-time delivery rate
        on_time = (self.df['Delivery_Time'] <= 30).sum() / len(self.df)
        print(f"On-time delivery rate (≤30 min): {on_time*100:.2f}%")
        
        # Average by restaurant type
        by_type = self.df.groupby('Restaurant_Type')['Delivery_Time'].mean()
        print(f"\nAverage delivery time by restaurant:\n{by_type}")
        
        # By time of day
        by_hour = self.df.groupby(self.df['Datetime'].dt.hour)['Delivery_Time'].mean()
        print(f"\nDelivery time by hour:\n{by_hour}")
```

## 🔧 Feature Engineering

```python
from datetime import datetime
import math

class DeliveryFeatureEngineer:
    """Create delivery features"""
    
    @staticmethod
    def distance_features(df):
        """Calculate distance metrics"""
        df_copy = df.copy()
        
        # Haversine distance if coordinates available
        if 'Restaurant_Latitude' in df_copy.columns:
            df_copy['Calculated_Distance'] = df_copy.apply(
                lambda row: haversine(
                    (row['Restaurant_Latitude'], row['Restaurant_Longitude']),
                    (row['Delivery_Latitude'], row['Delivery_Longitude'])
                ) if pd.notna(row['Restaurant_Latitude']) else np.nan,
                axis=1
            )
        
        return df_copy
    
    @staticmethod
    def temporal_features(df):
        """Extract time-based features"""
        df_copy = df.copy()
        
        df_copy['Datetime'] = pd.to_datetime(df_copy['Datetime'])
        df_copy['Hour'] = df_copy['Datetime'].dt.hour
        df_copy['DayOfWeek'] = df_copy['Datetime'].dt.dayofweek
        df_copy['Month'] = df_copy['Datetime'].dt.month
        
        # Peak hours (lunch 11-2, dinner 6-9)
        df_copy['IsPeakHour'] = df_copy['Hour'].isin([11, 12, 13, 18, 19, 20, 21]).astype(int)
        
        # Time to delivery (morning, afternoon, evening, night)
        df_copy['TimeOfDay'] = pd.cut(df_copy['Hour'], bins=[0, 12, 17, 21, 24],
                                      labels=[0, 1, 2, 3], right=False)
        
        return df_copy
    
    @staticmethod
    def order_features(df):
        """Order-level features"""
        df_copy = df.copy()
        
        # Number of items (proxy for order complexity)
        df_copy['Order_Items_Count'] = df_copy['Number_of_Items']
        
        # Order value impact
        df_copy['Order_Value_Category'] = pd.cut(df_copy['Order_Value'],
                                                  bins=[0, 250, 500, 1000, 5000],
                                                  labels=[0, 1, 2, 3])
        
        return df_copy
    
    @staticmethod
    def delivery_partner_features(df):
        """Partner performance metrics"""
        df_copy = df.copy()
        
        # Partner experience (number of deliveries)
        df_copy['Partner_Experience'] = df_copy['Partner_Number_of_deliveries']
        
        # Partner rating
        df_copy['Partner_Rating'] = df_copy['Delivery_partner_rating']
        
        # Partner age group
        df_copy['Partner_Age_Log'] = np.log1p(df_copy['Delivery_partner_age'])
        
        return df_copy
```

## 🤖 Regression Models

```python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

class DeliveryTimeRegressor:
    """Predict delivery time"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = self._build_models()
        self.best_model = None
    
    def _build_models(self):
        """Initialize models"""
        return {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=10.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
        }
    
    def train_all(self, X_train, y_train):
        """Train all models"""
        trained = {}
        
        X_scaled = self.scaler.fit_transform(X_train)
        
        for name, model in self.models.items():
            if name in ['Linear Regression', 'Ridge Regression']:
                model.fit(X_scaled, y_train)
            else:
                model.fit(X_train, y_train)
            
            trained[name] = model
        
        self.models = trained
        return trained
    
    def estimate_delivery_time(self, features):
        """Predict delivery time"""
        # Aggregate predictions from multiple models
        predictions = []
        
        for model in self.models.values():
            pred = model.predict(features.reshape(1, -1))
            predictions.append(pred[0])
        
        # Average prediction
        avg_time = np.mean(predictions)
        
        # Add buffer for reliability
        estimated_time = avg_time * 1.1  # 10% buffer
        
        return {
            'Prediction': avg_time,
            'Estimated_with_buffer': estimated_time,
            'Confidence_Range': (avg_time * 0.85, avg_time * 1.15)
        }
```

## 📊 Model Evaluation

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class DeliveryEvaluator:
    """Evaluate prediction models"""
    
    @staticmethod
    def regression_metrics(y_true, y_pred):
        """Calculate metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MAE (minutes)': round(mae, 2),
            'RMSE (minutes)': round(rmse, 2),
            'MAPE (%)': round(mape, 2),
            'R²': round(r2, 4)
        }
    
    @staticmethod
    def model_comparison(y_true, predictions_dict):
        """Compare models"""
        results = {}
        
        for model_name, y_pred in predictions_dict.items():
            results[model_name] = DeliveryEvaluator.regression_metrics(y_true, y_pred)
        
        comparison_df = pd.DataFrame(results).T
        print("\nModel Performance Comparison:")
        print(comparison_df.sort_values('R²', ascending=False))
        
        return comparison_df
    
    @staticmethod
    def prediction_distribution(y_true, y_pred):
        """Analyze prediction accuracy"""
        errors = y_true - y_pred
        
        print(f"Mean prediction error: {np.mean(errors):.2f} minutes")
        print(f"Std dev of errors: {np.std(errors):.2f} minutes")
        print(f"% predictions within ±5 min: {(np.abs(errors) <= 5).sum() / len(errors) * 100:.2f}%")
```

## 💡 Interview Talking Points

**Q: Key factors affecting delivery time?**
```
Answer:
- Distance to delivery location (most important)
- Time of day (peak hours slower)
- Order complexity (number of items)
- Delivery partner experience
- Weather/traffic conditions
- Restaurant preparation time
```

**Q: Handle outliers/unusual cases?**
```
Answer:
- Identify outliers (far deliveries, very long times)
- Separate model for them or robust loss
- Weather impact correction
- Traffic incident flagging
```

## 🌟 Portfolio Value

✅ Food delivery domain knowledge
✅ Geospatial distance calculations
✅ Time-series patterns
✅ Feature engineering (complex)
✅ Regression modeling
✅ Real-world delivery logistics
✅ Performance metrics

---

**Technologies**: Scikit-learn, Pandas, NumPy, Haversine

