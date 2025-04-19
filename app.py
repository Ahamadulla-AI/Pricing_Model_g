import pandas as pd
import joblib
import uvicorn
#import nest_asyncio
from fastapi import FastAPI
import logging
import os
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import Depends, HTTPException, status
import secrets
from src.Auth_services import get_current_username

from dotenv import load_dotenv
load_dotenv()


data_path = os.getenv("DATA_PATH", "data/Product_Pricing_api.csv")
model_path = os.getenv("MODEL_PATH", "models/RandomForest_price_predictor_20250417_220348.pkl")

log_dir = "logs"
#data_path = "data/Product_Pricing_api.csv"
#model_path = "models/RandomForest_price_predictor_20250417_220348.pkl"
# Setup logging: Create log directory and file
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "model_training_logs.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Enable nested event loops for Jupyter Notebooks or async environments
#nest_asyncio.apply()

app = FastAPI()

@app.get("/")
async def root(username: str = Depends(get_current_username)):
    return {"message": f"Welcome {username}, Pricing model API is up and running!"}

@app.get("/Pricing_Model")
async def predict_pricing(ProductID: str, ActualPrice: float, username: str = Depends(get_current_username)):
    try:
        logging.info("Received request with ProductID: %s, ActualPrice: %f", ProductID, ActualPrice)
        
        # Load dataset and compute additional columns needed for prediction
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
        df['LastUpdated'] = pd.to_datetime(df['LastUpdated'], format='%Y-%m-%d', errors='coerce')
        df['DaysSinceLastUpdate'] = (df['Date'] - df['LastUpdated']).dt.days
        logging.info("Computed DaysSinceLastUpdate.")

        # Get the list of unique product names for validation
        unique_ProductID = df['ProductID'].unique().tolist()
        if ProductID not in unique_ProductID:
            error_msg = f"Invalid ProductID '{ProductID}'. Expected one of: {unique_ProductID}"
            logging.error(error_msg)
            return {"error": error_msg}

        # Validate ActualPrice within allowed range from dataset
        actual_min_price = df['ActualPrice'].min()
        actual_max_price = df['ActualPrice'].max()
        if ActualPrice < actual_min_price or ActualPrice > actual_max_price:
            error_msg = f"Entered ActualPrice {ActualPrice} is out of range. Allowed range: {actual_min_price} - {actual_max_price}"
            logging.error(error_msg)
            return {"error": error_msg}

        # Drop only unnecessary columns, but keep those needed by the model pipeline
        drop_columns = ['SuggestedPrice', 'ProductName', 'Date', 'LastUpdated', 'id', 'SupplierID']
        df_dropped = df.drop(drop_columns, axis=1, errors='ignore')
        logging.info("Remaining columns for inference: %s", df_dropped.columns.tolist())

        # Identify numerical (excluding ActualPrice) and categorical columns
        numerical_cols = df_dropped.select_dtypes(include=['int64', 'float64']).columns.difference(['ActualPrice'])
        categorical_cols = df_dropped.select_dtypes(include=['object', 'category']).columns

        # Calculate mean for numerical features and mode for categorical features
        mean_values = df_dropped[numerical_cols].mean().to_dict()
        mode_values = {col: df_dropped[col].mode().iloc[0] for col in categorical_cols}
        logging.info("Calculated mean values: %s", mean_values)
        logging.info("Calculated mode values: %s", mode_values)

        # Construct the new sample using user input plus the calculated values
        new_sample = {'ProductID': ProductID, 'ActualPrice': ActualPrice}
        new_sample.update(mean_values)
        new_sample.update(mode_values)
        logging.info("Constructed sample data for prediction (before NaN conversion): %s", new_sample)

        # Convert any NaN values to None (JSON null)
        new_sample = {k: (None if pd.isna(v) else v) for k, v in new_sample.items()}
        logging.info("Sample data after converting NaN values: %s", new_sample)

        # Create a DataFrame ensuring column order matches the training dataset
        User_data = pd.DataFrame([new_sample])
        User_data = User_data[df_dropped.columns]
        logging.info("Final user data for prediction: %s", User_data.to_dict(orient='records'))


        
        # Load the pre-trained model pipeline (with preprocessing)
        loaded_model = joblib.load(model_path)
        logging.info("Loaded model from: %s", model_path)
        
        # Predict using the loaded model pipeline
        prediction = loaded_model.predict(User_data)
        logging.info("Prediction result: %s", prediction.tolist())
        
        return {"ProductID": ProductID, "ActualPrice": ActualPrice, "Prediction": prediction.tolist()} ##, "InputData": new_sample}
    except Exception as e:
        logging.error("An error occurred during prediction: %s", e)
        return {"error": str(e)}

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7777)


## ngrok http http://127.0.0.1:7777
