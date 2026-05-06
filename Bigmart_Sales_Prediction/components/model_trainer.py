import os
import sys
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

from Bigmart_Sales_Prediction.exception.exception import CustomException
from Bigmart_Sales_Prediction.logging.logger import logging

from Bigmart_Sales_Prediction.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from Bigmart_Sales_Prediction.entity.config_entity import ModelTrainerConfig

from Bigmart_Sales_Prediction.utils.ml_utils.model.estimator import CarPriceModel
from Bigmart_Sales_Prediction.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models,
)
from Bigmart_Sales_Prediction.utils.ml_utils.metric.classification_metric import (
    get_classification_score,
)

from urllib.parse import urlparse


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)


    # -------------------------
    # Train Model
    # -------------------------
    def train_model(self, X_train, y_train, X_test, y_test):
        try:
        

        # -------------------------
        # Models
        # -------------------------
            models = {
                "LinearRegression": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor()
            }

        # -------------------------
        # Hyperparameters (REGRESSION)
        # -------------------------
            params = {

                "LinearRegression": {
                    "fit_intercept": [True, False]
                },

                "KNN": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2]
                },

                "DecisionTree": {
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                },

                "RandomForest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                }
            }

        # -------------------------
        # Evaluate Models
        # -------------------------
            model_report, trained_models = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

        # -------------------------
        # Select Best Model (LOWEST RMSE)
        # -------------------------
            best_model_name = min(model_report, key=model_report.get)
            best_model = trained_models[best_model_name]
            best_model_score = model_report[best_model_name]

            logging.info(f"Best Model: {best_model_name}, RMSE: {best_model_score}")

        # -------------------------
        # Train Metrics
        # -------------------------
       

            y_train_pred = best_model.predict(X_train)

            train_metrics = {
                "rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
                "mae": mean_absolute_error(y_train, y_train_pred),
                "r2": r2_score(y_train, y_train_pred)
            }

        # -------------------------
        # Test Metrics
        # -------------------------
            y_test_pred = best_model.predict(X_test)

            test_metrics = {
                "rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
                "mae": mean_absolute_error(y_test, y_test_pred),
                "r2": r2_score(y_test, y_test_pred)
            }

        # -------------------------
        # Load Preprocessor
        # -------------------------
            preprocessor = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )

        # -------------------------
        # Save Final Model (Combined)
        # -------------------------
            try:
                model_dir_path = os.path.dirname(
                    self.model_trainer_config.trained_model_file_path
                )
                os.makedirs(model_dir_path, exist_ok=True)

                Car_Price_Model = CarPriceModel(  # 🔥 Rename this class ideally
                    preprocessor=preprocessor,
                    model=best_model,
                )

            except Exception as e:
                logging.info("Unable to create model.pkl")
                raise CustomException(e, sys)

            logging.info("Saving final combined model...")

            save_object(
                self.model_trainer_config.trained_model_file_path,
                Car_Price_Model
            )

        # Optional: Save raw model separately
            os.makedirs("final_model", exist_ok=True)
            save_object("final_model/model.pkl", best_model)

        # -------------------------
        # Return Artifact
        # -------------------------
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metrics,
                test_metric_artifact=test_metrics,
            )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------
    # Entry Point
    # -------------------------
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            return self.train_model(X_train, y_train, X_test, y_test)

        except Exception as e:
            raise CustomException(e, sys)