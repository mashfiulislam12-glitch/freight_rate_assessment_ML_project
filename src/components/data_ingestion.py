import os 
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts','train_data.csv')
    test_data_path=os.path.join('artifacts','test_data.csv')
    raw_data_path=os.path.join('artifacts','data.csv')

class DataIngestion:
    try:

        def __init__(self):
            self.dataingestionconfig=DataIngestionConfig()

        def initiate_data_ingestion(self):
            logging.info('initializing data ingestion')
            df=pd.read_csv('src/notebook/data/train-test.csv')

            os.makedirs(os.path.dirname(self.dataingestionconfig.train_data_path),exist_ok=True)

            df.to_csv(self.dataingestionconfig.raw_data_path,index=False,header=True)

            train_data,test_data=train_test_split(df,test_size=0.2,random_state=42)

            train_data.to_csv(self.dataingestionconfig.train_data_path,index=False,header=True)
            test_data.to_csv(self.dataingestionconfig.test_data_path,index=False,header=True)

            logging.info('ingestion has been completed')

            return(
                self.dataingestionconfig.train_data_path,
                self.dataingestionconfig.test_data_path
            )
        
    except Exception as e:
        raise CustomException(e,sys)


if __name__=='__main__':
    obj=DataIngestion()
    train_path,test_path=obj.initiate_data_ingestion()

    print('train_path:',train_path)
    print('test_path:',test_path)

    obj = DataTransformation()

    X_train_arr, y_train, X_test_arr, y_test = (
        obj.initiate_data_transformation(
            train_path,
            test_path
        )
    )

    obj=ModelTrainer()
    obj.initiate_model_trainer(X_train_arr, y_train, X_test_arr, y_test)
