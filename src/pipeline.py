import pandas as pd
import os
import sys
import json
from from_root import from_root
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold

from src.logger import logging
from src.exception import ElectricityException

def get_data(filename: str) -> pd.DataFrame:
    try:
        # Load the data
        filepath = os.path.join(from_root(), filename)
        data = pd.read_csv(filepath)
        return data
    except Exception as e:
        raise ElectricityException(e, sys)


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    try:
        # Preprocess the data
        label_encoder = LabelEncoder()
        data['sectorName'] = label_encoder.fit_transform(data['sectorName'])
        data['stateDescription'] = label_encoder.fit_transform(data['stateDescription'])
        data.drop(['customers', 'revenue', 'sales'], axis=1, inplace=True)
        return data
    except Exception as e:
        raise ElectricityException(e, sys)


def model_trainer(data: pd.DataFrame) -> RandomForestRegressor:
    try:
        # Split the data
        X = data.drop(['price'], axis=1)
        y = data['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Hyperparameter tuning
        param_grid = {'n_estimators': [50, 100, 150, 200, 250]}
        
        rf = RandomForestRegressor(random_state=42)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Get the best number of estimators
        best_n_estimators = grid_search.best_params_['n_estimators']

        # Train the final model
        best_rf = RandomForestRegressor(n_estimators=best_n_estimators, random_state=42)
        best_rf.fit(X_train, y_train)

        # Predict and evaluate
        y_pred_test = best_rf.predict(X_test)

        mse_test = mean_squared_error(y_test, y_pred_test)

        print("Best number of estimators:", best_n_estimators)
        print("Mean Squared Error (MSE) on test set:", mse_test)

        scores = {"Best_n_estimators": best_n_estimators, "MSE_Score": mse_test}
        with open("scores.json","w") as scoresfile:
            json.dump(scores, scoresfile)

        return best_rf
    except Exception as e:
        raise ElectricityException(e, sys)


def save_model(model: RandomForestRegressor):
    try:
        # Save the model
        with open('model.pkl', 'wb') as file:
            pickle.dump(model, file)
    except Exception as e:
        raise ElectricityException(e, sys)


def pipeline_run():
    try:
        logging.info("Pipeline started")
        csv_filename = "electricity_data.csv"
        
        data = get_data(csv_filename)
        logging.info("Data loaded successfully")

        logging.info("Data preprocessing Started")
        data = preprocess_data(data)
        logging.info("Data preprocessing completed")
        
        logging.info("Training and evaluation started")
        model = model_trainer(data)
        logging.info("Training and evaluation completed")

        save_model(model)
        logging.info("Model saved successfully")

        logging.info("Pipeline completed successfully")
    except ElectricityException as e:
        logging.error(f"Pipeline failed due to: {str(e)}")
        raise ElectricityException(e, sys)


if __name__ == '__main__':
    pipeline_run()
