import pandas as pd
import os
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import MyException
from src.logger import logging
from src.utils import evaluate_model, file_save


@dataclass
class ModelTrainerConfig:
    root_folder = "./artifacts/"


class ModelTrainer:
    def __init__(self, target_variable="math score") -> None:
        self.model_trainer_config = ModelTrainerConfig()
        self.target_variable = target_variable

    def initiate_model_training(self):
        training_df = pd.read_csv(os.path.join(self.model_trainer_config.root_folder,'train_transformed_df.csv'))
        testing_df = pd.read_csv(os.path.join(self.model_trainer_config.root_folder,'test_transformed_df.csv'))
        
        x_train, y_train, x_test, y_test = (
            training_df.iloc[:,:-1],
            training_df.iloc[:,-1],
            testing_df.iloc[:,:-1],
            testing_df.iloc[:,-1],
        )

        models = {
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression(),
            "XGBRegressor": XGBRegressor(),
            # "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor(),
        }
        params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

        evaluation_report = evaluate_model(
            model_dict=models,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            params=params
        )

        best_model = max(evaluation_report, key=lambda k: evaluation_report[k])
        best_model_result = evaluation_report[best_model]

        logging.debug(f"the best model is {best_model}")
        logging.info(f"The r2 score for the best model is {best_model_result}")

        file_save(
            path=self.model_trainer_config.root_folder,
            title="trained_best_model.pkl",
            artifact=best_model,
            model=True,
        )
        
        logging.info(f"The file is saved")
        
        return best_model_result
    
    
