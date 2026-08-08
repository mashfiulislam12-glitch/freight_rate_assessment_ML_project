import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.utils import evaluate_models
from src.utils import save_object




import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)





@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
        def __init__(self):
            self.modeltrainerconfig=ModelTrainerConfig()

        def initiate_model_trainer(self,X_train_arr,y_train,X_test_arr,y_test):
            try:
                models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Decision Tree": DecisionTreeRegressor(),
                #"Random Forest": RandomForestRegressor(),
                #"Extra Trees": ExtraTreesRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False)
                }


                model_report: dict = evaluate_models(

                        X_train=X_train_arr,

                        y_train=y_train,

                        X_test=X_test_arr,

                        y_test=y_test,

                        models=models

                    )

                best_model_score = max(model_report.values())

                best_model_name = list(model_report.keys())[

                        list(model_report.values()).index(best_model_score)]
                print(best_model_name)

                best_model = models[best_model_name]
                
                best_model.fit(X_train_arr, y_train)
                predicted=best_model.predict(X_test_arr)

                R2_score=r2_score(y_test,predicted)

                save_object(
                    file_path=self.modeltrainerconfig.trained_model_file_path,
                    obj=best_model
                )
                print(R2_score)
                return R2_score



            except Exception as e:
                raise CustomException(e,sys)