import sys
import pandas as pd

from src.logger import logging
from src.utils import load_object
from src.exception import MyException
class PredictionPipeline():
    def __init__(self) -> None:
        pass
    def prediction(self,features):
        try:
            model_path = './artifacts/trained_best_model.pkl'
            preprocessor_path = './artifacts/preprocessor.pkl'
            logging.info('Loaded in the necessities')
            model = load_object(filepath = model_path)
            preprocesor = load_object(filepath = preprocessor_path)
            print(type(preprocesor))
            features_scaled = preprocesor.transform(features)
            preds = model.predict(features_scaled)
            return preds
        except Exception as e:
            raise MyException(e,sys)
        
class DataCreation():
    def __init__( self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score
    
    def get_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise MyException(e, sys)
        