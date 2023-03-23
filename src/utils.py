import pandas as pd
from src.logger import logging
import os
    
    
def file_save(path,df,title):
    logging.info(f'saved the file: {title}')
    df.to_csv(os.path.join(path,str(title)),index=False)
    return