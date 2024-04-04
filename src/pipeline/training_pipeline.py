import sys 
import os

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
class Training_pipeline():
    def __init__(self) -> None:
        pass

    def Training(self, dataframe):
        try:    
            logging.info("training initiated")
            data_intake = DataIngestion()
            data_transform = DataTransformation()
            trainer = ModelTrainer()

            train_path, test_path = data_intake.initiate_data_ingestion(df=dataframe)
            train_arr, test_arr,_ = data_transform.initiate_data_transformation(train_path=train_path, test_path=test_path)
            r2s = trainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr)

            return (r2s*100)

        except Exception as e:
            CustomException(e, sys)