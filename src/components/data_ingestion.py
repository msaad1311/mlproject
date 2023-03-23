import os
import pandas as pd
import sys

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import MyException
from src.logger import logging
from src.utils import file_save
@dataclass
class DataIngestionConfig():
    root_folder = r'artifacts/'  
    
class DataIngestion():
    def __init__(self) -> None:
        self.data_ingestion_config = DataIngestionConfig()
        
    # def file_save(self,df,title):
    #     logging.info(f'saved the file: {title}')
    #     df.to_csv(os.path.join(self.data_ingestion_config.root_folder,str(title)),index=False)
    #     return
                  
    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv(r'data/StudentsPerformance.csv')
            logging.info('Read the raw data')
            training_data, testing_data = train_test_split(df,test_size=0.2,random_state=42)
            logging.info('Created the train test split')
            os.makedirs(self.data_ingestion_config.root_folder,exist_ok = True)
            logging.info('Made the directory')
            df_to_save = {'raw_data.csv':df,
                          'train_data.csv':training_data,
                          'test_data.csv':testing_data}
            
            for key,value in df_to_save.items():
                # print(key,value)
                file_save(path=self.data_ingestion_config.root_folder,df=value,title=key)
            logging.info('Saved the files')
            return (
                self.data_ingestion_config.root_folder
            )
            
        except Exception as e:
            raise MyException(e,sys)
        
        finally:
            logging.info('Completed the ingestion phase')
        

if __name__=="__main__":
    obj = DataIngestion()
    placeholder = obj.initiate_data_ingestion()
            
            