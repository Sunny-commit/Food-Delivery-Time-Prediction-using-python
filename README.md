
# 🍔 Food Delivery Time Prediction Using Python

This project predicts the time it takes to deliver food based on various factors such as restaurant and delivery location coordinates, delivery person details, and more. It leverages feature engineering using geographic coordinates and applies regression models to forecast delivery duration.

---

## 📁 Dataset

The dataset (`deliverytime.txt`) includes the following features:

* `Restaurant_latitude`, `Restaurant_longitude`
* `Delivery_location_latitude`, `Delivery_location_longitude`
* `Delivery_person_Age`
* `Delivery_person_ratings`
* `time_taken(min)` — **Target variable**

---

## 🧠 Key Feature Engineering

* **Haversine Formula** is used to calculate the actual distance between the restaurant and the delivery location based on latitude and longitude.
* The calculated distance becomes a crucial feature in predicting delivery time.

```python
# Haversine Formula used in the notebook:
def distcalculate(lat1, lon1, lat2, lon2):
    ...
```

---

## 🧰 Libraries Used

* **Pandas** and **NumPy** – for data wrangling
* **Matplotlib** and **Plotly Express** – for data visualization
* **Scikit-learn / Statsmodels** – for regression modeling and performance analysis

---

## 📊 Exploratory Data Analysis (EDA)

Visualizations show how delivery time is influenced by:

* Distance
* Age of the delivery person
* Delivery person ratings

Interactive scatter plots with trendlines are used to identify linear relationships.

---

## 🧪 Model Building

* A **Linear Regression model** is trained to predict `time_taken(min)` using:

  * Distance
  * Ratings
  * Age
* Trendlines help validate the linearity assumption before model training.

---

## 🧩 How to Run

1. Clone the repository.
2. Ensure the dataset file `deliverytime.txt` is in the project directory.
3. Run the notebook using:

   ```bash
   jupyter notebook "Food Delivery Time Prediction using python.ipynb"
   ```
4. Experiment with new inputs to test the model's prediction capability.

---

## 🔮 Possible Extensions

* Include weather or traffic data to improve predictions.
* Use non-linear regression or ensemble models for better performance.
* Deploy using a simple Flask or Streamlit web interface.

---

## 🏁 Folder Structure

```
📦Food Delivery Time Prediction
 ┣ 📜Food Delivery Time Prediction using python.ipynb
 ┣ 📜deliverytime.txt
 ┗ 📜README.md
```
