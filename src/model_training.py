# -*- coding: utf-8 -*-
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

import logging
import os

# Define the log directory and filename
log_dir = r"D:\EUR_AI_MASTER_PROJECTS\AI_Based_Product_Pricing\logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "model_training_logs.log")

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def text_extractor(x):
    """
    Convert the input series to a 1D numpy array.
    """
    logging.info("Text extractor started")
    print("Text extractor started")

    return x.to_numpy()

class PricePredictor:
    def __init__(self, model_choice='RandomForest', model_dir='models'):
        """
        Initialize the PricePredictor with model selection and create the preprocessor.
        The trained models will be stored in the specified directory.
        """
        logging.info("PricePredictor initialized with model choice: %s", model_choice)
        print("PricePredictor initialized with model choice:", model_choice)
        # Set the model choice and directory
        self.model_choice = model_choice
        self.model_dir = model_dir
        self.model = None
        self.preprocessor = None
        self.pipeline = None
        self.numeric_features = ['ActualPrice', 'PurchasePrice', 'Additional_Cost', 
                                   'Tax_Duties', 'TotalCost', 'CurrentStock', 'LeadTime_days',
                                   'CompetitorPrice', 'Past3MonthsDemand', 'DaysSinceLastUpdate']
        self.ordinal_features = ['SeasonalityFactor', 'MarketTrendFactor']
        self.text_feature = "ProductName"  # TF-IDF processing column
        self.categorical_features = ['Region', 'CompetitorName']  # non-text categoricals
        # Define ordinal orders
        self.seasonality_order = ['Low', 'Medium', 'High']
        self.market_trend_order = ['Decreasing', 'Stable', 'Increasing']
        
        self._create_preprocessor()

        # Ensure the model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
    
    def _create_preprocessor(self):
        """
        Create a ColumnTransformer for preprocessing:
         - Scale numeric features.
         - Encode ordinal features.
         - Process text using TF-IDF.
         - One-hot encode other categorical features.
        """
        logging.info("Preprocessor creation started")
        print("Preprocessor creation started")
        # Define the text processing pipeline
        try:
            text_pipeline = Pipeline([
                ('extract', FunctionTransformer(text_extractor, validate=False)),
                ('tfidf', TfidfVectorizer(max_features=1000))
            ])

            self.preprocessor = ColumnTransformer(transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('ord', OrdinalEncoder(categories=[self.seasonality_order, self.market_trend_order]), self.ordinal_features),
                ('txt', text_pipeline, self.text_feature),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ])
        except Exception as e:
            logging.info("Error creating preprocessor:", e)
            print("Error creating preprocessor:", e)
            raise

    def load_data(self, file_path):
        """
        Load data from CSV and perform initial cleaning and date handling.
        """
        logging.info("Data loading started")
        print("Data loading started")
        try:
            df = pd.read_csv(file_path)
            # Use explicit date format (adjust format if needed)
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
            df['LastUpdated'] = pd.to_datetime(df['LastUpdated'], format='%Y-%m-%d', errors='coerce')
            df['DaysSinceLastUpdate'] = (df['Date'] - df['LastUpdated']).dt.days
            df.drop(['Date', 'LastUpdated'], axis=1, inplace=True)
            return df
        except Exception as e:
            logging.info("Error loading data:", e)
            print("Error loading data:", e)
            raise

    def prepare_data(self, df):
        """
        Splits the dataframe into features and target, then into training and testing sets.
        """
        logging.info("Data preparation started")
        print("Data preparation started")
        try:
            X = df.drop(['SuggestedPrice', 'ProductID'], axis=1)  # drop identifiers and target
            y = df['SuggestedPrice']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.info("Error preparing data:", e)
            print("Error preparing data:", e)
            raise

    def _get_model(self):
        """
        Returns the chosen model.
        """
      
        logging.info("Model selection started")
        print("Model selection started")

        if self.model_choice == 'RandomForest':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_choice == 'GradientBoosting':
            return GradientBoostingRegressor(random_state=42)
        elif self.model_choice == 'LinearRegression':
            return LinearRegression()
        else:
            raise ValueError("Model choice not recognized. Choose among 'RandomForest', 'GradientBoosting', or 'LinearRegression'.")

    def train(self, X_train, y_train):
        """
        Build the pipeline, train the model, and log metrics with MLflow.
        """
        logging.info("Model training started")
        print("Model training started")
        try:
            self.model = self._get_model()
            self.pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('regressor', self.model)
            ])
            # Set an experiment. This creates a new experiment if it doesn't exist.
            mlflow.set_experiment("PricePredictionExperiment")
            with mlflow.start_run():
                self.pipeline.fit(X_train, y_train)
                mlflow.sklearn.log_model(self.pipeline, "model")
                logging.info("Model training completed and logged with MLflow.")
                print("Model training completed and logged with MLflow.")
        except Exception as e:
            logging.info("Error during training:", e)
            print("Error during training:", e)
            raise

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test set and log evaluation metrics with MLflow.
        """

        logging.info("Model evaluation started")
        print("Model evaluation started")
        try:
            y_pred = self.pipeline.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            with mlflow.start_run(nested=True):
                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("R2", r2)
            print("Evaluation Metrics:")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R2: {r2:.4f}")
            return {"RMSE": rmse, "MAE": mae, "R2": r2}
        except Exception as e:
            logging.info("Error during evaluation:", e)
            print("Error during evaluation:", e)
            raise

    def retrain(self, new_data_df):
        """
        Retrain the model on new data, evaluate it, and save the model with versioning (using timestamp).
        """
        try:
            X_train, X_test, y_train, y_test = self.prepare_data(new_data_df)
            self.train(X_train, y_train)
            metrics = self.evaluate(X_test, y_test)
            self.save_model()  # This saves the model with a timestamp in the file name.
            return metrics
        except Exception as e:
            logging.info("Error during retraining:", e)
            print("Error during retraining:", e)
            raise

    def save_model(self):
        """
        Save the final pipeline with a timestamp appended to the file name.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_file = os.path.join(self.model_dir, f"{self.model_choice}_price_predictor_{timestamp}.pkl")
            joblib.dump(self.pipeline, model_file)
            logging.info("Model saved to: %s", model_file)
            print("Model saved to:", model_file)
        except Exception as e:
            logging.info("Error saving model:", e)
            print("Error saving model:", e)
            raise

    def load_model(self, model_file):
        """
        Load a previously saved model pipeline.
        """
        try:
            self.pipeline = joblib.load(model_file)
            logging.info("Model loaded from: %s", model_file)
            print("Model loaded from:", model_file)
        except Exception as e:
            logging.info("Error loading model:", e)
            print("Error loading model:", e)
            raise

    def predict(self, sample_data):
        """
        Make predictions on sample data.
        """
        logging.info("Make predictions on sample data is started")
        print("Make predictions on sample data is started")
        try:
            prediction = self.pipeline.predict(sample_data)
            return prediction
        except Exception as e:
            logging.info("Error during prediction:", e)
            print("Error during prediction:", e)
            raise

    def feature_importance(self, X_train, feature_plot=True):
        """
        Extract and optionally plot feature importances for models that provide them.
        """
        logging.info("Feature_Importance for model started")
        print("Feature_Importance for model started")
        try:
            self.preprocessor.fit(X_train)
            num_features = self.numeric_features
            tfidf_features = self.pipeline.named_steps['preprocessor']\
                                .named_transformers_['txt']\
                                .named_steps['tfidf'].get_feature_names_out()
            cat_features = list(self.pipeline.named_steps['preprocessor']\
                                .named_transformers_['cat'].get_feature_names_out(self.categorical_features))
            ord_features = self.ordinal_features
            feature_names = list(num_features) + list(tfidf_features) + list(cat_features) + list(ord_features)
            
            regressor = self.pipeline.named_steps['regressor']
            if hasattr(regressor, 'feature_importances_'):
                importances = regressor.feature_importances_
                if feature_plot:
                    plt.figure(figsize=(12, 8))
                    sns.barplot(x=importances, y=feature_names)
                    plt.title('Feature Importances')
                    plt.tight_layout()
                    plt.show()
                return dict(zip(feature_names, importances))
            else:
                logging.info("The model does not provide feature importances.")
                print("The model does not provide feature importances.")
                return None
        except Exception as e:
            logging.info("Error extracting feature importances:", e)
            print("Error extracting feature importances:", e)
            raise