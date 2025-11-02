Customer Purchase Prediction using XGBoost + FastAPI
🛍️ Overview

This project predicts the purchase amount of retail customers based on their demographics and product details.
It demonstrates the end-to-end ML pipeline — from EDA → Feature Engineering → Model Building → FastAPI Deployment — in a clean, reproducible workflow.

🔍Workflow Summary
1️⃣ Exploratory Data Analysis (EDA)

Analysed customer demographics and product-level trends.

Visualised purchase patterns by age, gender, city type, and product category.

Identified and handled missing values and outliers.

2️⃣ Feature Engineering

Encoded categorical features: Gender, City_Category, Stay_In_Current_City_Years.

Cleaned and aligned training/test datasets.

Scaled features and ensured numeric consistency for model training.

3️⃣ Model Development
Model	RMSE	R²	Remarks
Linear Regression	4,674	0.13	Weak linear fit
XGBoost	2,895	0.666	Strong nonlinear performance

Product_Category_1 was found to be the most influential predictor.

Saved the trained model and column mapping using joblib and json.

4️⃣ FastAPI Deployment

Built a FastAPI service to expose the trained model via REST API:

POST /predict → Predicts purchase amount for a single record

POST /predict_batch → Handles multiple records

Model loads once at startup for fast inference.

Tested endpoints via Swagger UI (/docs) and curl commands.

⚙️ Tech Stack

Languages / Frameworks: Python, FastAPI

Libraries: Pandas, NumPy, Scikit-learn, XGBoost, Joblib

Server: Uvicorn

(Optional for deployment) Docker, Render / Azure App Service

🚀 Results

✅ XGBoost outperformed linear models — reduced RMSE by ~38%
✅ Explained 67% of purchase variance, showing strong model generalisation
✅ Fully functional REST API ready for cloud deployment

🧾 Example API Request
POST /predict
{
  "data": {
    "Gender": 1,
    "Age": 3,
    "Marital_Status": 0,
    "Occupation": 7,
    "Stay_In_Current_City_Years": 3,
    "B": 0,
    "C": 1,
    "Product_Category_1": 5,
    "Product_Category_2": 8,
    "Product_Category_3": 12
  }
}

Response
{"prediction": 7171.36}

🧩 How to Run Locally
# 1️⃣ Install dependencies
pip install -r requirements.txt

# 2️⃣ Start FastAPI server
uvicorn app:app --reload

# 3️⃣ Open browser for interactive docs
http://127.0.0.1:8000/docs
