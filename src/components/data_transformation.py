import pandas as pd
from src.exception import MyException
from src.logger import logging
import os
import sys
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataclasses import dataclass
from src.utils import file_save


@dataclass
class DataTransformationConfig:
    root_folder = "./artifacts/"


class DataTransformation:
    def __init__(self, target_variable="math score") -> None:
        self.data_transformation_config = DataTransformationConfig()
        self.target_variable = target_variable

    def get_transformer_object(self):
        try:
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoding", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            return num_pipeline, cat_pipeline
        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_transformation(self):
        try:
            train_df = pd.read_csv(
                os.path.join(
                    self.data_transformation_config.root_folder, "train_data.csv"
                )
            )
            test_df = pd.read_csv(
                os.path.join(
                    self.data_transformation_config.root_folder, "test_data.csv"
                )
            )

            logging.info(f"Read both training and testing data")
            logging.debug(f"Training data shape is {train_df.shape}")
            logging.debug(f"Testing data shape is {test_df.shape}")

            numerical_cols = list(
                train_df.select_dtypes(include=["int", "float"]).columns
            )
            numerical_cols.remove(self.target_variable)
            categorical_cols = list(train_df.select_dtypes(include=["object"]).columns)

            logging.debug(f"Numerical columns are: {numerical_cols}")
            logging.debug(f"Categorical columns are: {categorical_cols}")

            num_pipeline, cat_pipeline = self.get_transformer_object()

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_cols),
                    ("cat_pipeline", cat_pipeline, categorical_cols),
                ]
            )

            train_features_df = train_df.drop(columns=[self.target_variable], axis=1)
            train_target_df = train_df[self.target_variable]

            test_features_df = test_df.drop(columns=[self.target_variable], axis=1)
            test_target_df = test_df[self.target_variable]

            logging.info("splitted the dataframes to feature and target sections")

            feature_train = preprocessor.fit_transform(train_features_df)
            feature_test = preprocessor.transform(test_features_df)

            train_transformed_df = pd.DataFrame(
                np.c_[feature_train, np.array(train_target_df)]
            )
            test_transformed_df = pd.DataFrame(
                np.c_[feature_test, np.array(test_target_df)]
            )

            logging.info("Performed the transformation")

            file_save(
                path=self.data_transformation_config.root_folder,
                title="train_transformed_df.csv",
                artifact=train_transformed_df,
            )
            file_save(
                path=self.data_transformation_config.root_folder,
                title="test_transformed_df.csv",
                artifact=test_transformed_df,
            )

            logging.info("Saved the files")

            return self.data_transformation_config.root_folder

        except Exception as e:
            raise MyException(e, sys)


if __name__ == "__main__":
    x = DataTransformation()
    x.initiate_data_transformation()
