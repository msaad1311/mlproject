import pandas as pd
import sys
from src.logger import logging
from src.exception import MyException
import os
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def file_save(path, artifact, title, model=False):
    try:
        logging.info(f"saved the file: {title}")
        if model:
            with open(os.path.join(path, str(title)), "wb") as file_obj:
                dill.dump(model, file_obj)
        else:
            artifact.to_csv(os.path.join(path, str(title)), index=False)
        return
    except Exception as e:
        raise MyException(e, sys)


def evaluate_model(model_dict, x_train, x_test, y_train, y_test,params):
    try:
        result = {}
        for model_name, model in model_dict.items():
            logging.info(f"Working on {model_name}")
            
            param = params[model_name]
            
            gs = GridSearchCV(model,param,cv=3)
            gs.fit(x_train,y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)
            
            y_predicted = model.predict(x_test)
            r2_result = r2_score(y_test, y_predicted)

            result[model] = r2_result

        return result
    except Exception as e:
        raise MyException(e, sys)
