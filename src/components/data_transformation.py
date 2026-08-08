import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src.utils import save_object





@dataclass
class DataTransformationConfig:
    cleaner_obj_file_path=os.path.join('artifacts','cleaner.pkl')
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.datatransformationconfig=DataTransformationConfig()

    def get_data_transformation_obj(self):
        logging.info('starting data transformation')
        try:
            numerical_features=['pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon', 'distance',
            'weight', 'market_index', 'quote_signal', 'year',
            'month', 'day', 'day_of_week', 'week_of_year']
            categorical_features=['pickup', 'delivery', 'equipment']

            preprocessor= ColumnTransformer(
                transformers=[
                    ("num",SimpleImputer(strategy="median"),numerical_features,),
                    ("cat",Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore"))]),
                        categorical_features)])

            return preprocessor
            
        except Exception as e :
            raise CustomException(e,sys)


    def clean_data(self,df):
        try:
            df=df.drop_duplicates()
            df.loc[df["weight"] < 0, "weight"] = np.nan
            df['date']=pd.to_datetime(df['date'])
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["day"] = df["date"].dt.day
            df["day_of_week"] = df["date"].dt.dayofweek
            df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
            df.drop(columns=['load_id','date'],inplace=True)
        
            return df
        except Exception as e:
            CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)
            cleaner_obj=self.clean_data
            train_data=cleaner_obj(train_data)
            test_data=cleaner_obj(test_data)

            preprocessor_obj=self.get_data_transformation_obj()
    

            target_column_name='posted_rate'

            X_train=train_data.drop(target_column_name,axis=1)
            y_train=train_data[target_column_name]

            X_test=test_data.drop(target_column_name,axis=1)
            y_test=test_data[target_column_name]

            X_train_arr=preprocessor_obj.fit_transform(X_train)
            X_test_arr=preprocessor_obj.transform(X_test)


            save_object(
                file_path=self.datatransformationconfig.cleaner_obj_file_path,
                obj=cleaner_obj
            )
            save_object(
                file_path=self.datatransformationconfig.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return(
                X_train_arr,
                y_train,
                X_test_arr,
                y_test
            )

        except Exception as e:
            CustomException(e,sys)
