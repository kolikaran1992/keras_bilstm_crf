import logging
from keras_NER.__paths__ import path_to_logs
import datetime
from keras_NER.__common__ import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s')

file_handler = logging.FileHandler(path_to_logs.joinpath(datetime.datetime.now().strftime("%Y-%m-%d")).as_posix())
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.setLevel(logging.INFO)
