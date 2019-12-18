from nltk.tokenize import RegexpTokenizer
from unidecode import unidecode
from pathlib import Path
import json
from keras.models import model_from_json
from keras_contrib.layers import CRF


def get_word_tokenizer():
    return RegexpTokenizer(r'\w+|[^\w\s]').tokenize

def process_text(text):
    return unidecode(text)

def validate_path(path, logger):
    if not isinstance(path, Path):
        path = Path(path)

    if not path.is_file():
        logger.error('{} is not a valid file'.format(path.as_posix()))
        exit(101)

def read_label_file(path, logger):
    validate_path(path, logger)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip().split(' ')


def save_model(model, weights_file, params_file):
    with open(params_file, 'w') as f:
        params = model.to_json()
        json.dump(json.loads(params), f, sort_keys=True, indent=4)
        model.save_weights(weights_file)

def load_model(weights_file, params_file):
    with open(params_file) as f:
        model = model_from_json(f.read(), custom_objects={'CRF': CRF})
        model.load_weights(weights_file)

    return model
