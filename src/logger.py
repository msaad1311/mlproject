import logging
import sys
import os

from datetime import datetime

class CustomFormatter(logging.Formatter):

    cyan = "\u001b[36m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)" # type: ignore

    FORMATS = {
        logging.DEBUG: cyan + format + reset, # type: ignore
        logging.INFO: grey + format + reset, # type: ignore
        logging.WARNING: yellow + format + reset, # type: ignore
        logging.ERROR: red + format + reset, # type: ignore
        logging.CRITICAL: bold_red + format + reset # type: ignore
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)


logging.basicConfig(
    filename= LOG_FILE_PATH,
    level=logging.DEBUG,
    format= '[%(asctime)s] %(name)s %(levelname)s - %(message)s'
    
)
# set up logging to console
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(CustomFormatter())
# add the handler to the root logger
logging.getLogger('').addHandler(console)
