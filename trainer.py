from pathlib import Path
from nltk.tokenize import RegexpTokenizer
from functools import reduce

from keras_NER.__common__ import LOGGER_NAME
import logging
logger = logging.getLogger(LOGGER_NAME)

from keras_NER.wrapper import ModelWrapper

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--path_train", dest="path_train", default=None,
                    help="path to training data", required=True)
parser.add_argument("--path_valid", dest="path_valid", default=None,
                    help="path to validation data", required=True)

parser.add_argument("--path_w2v", dest="path_w2v", default=None,
                    help="path to word2vec keyed vectors", required=True)
parser.add_argument("--word_emb_dim", dest="word_emb_dim", default=100,
                    help="word embedding dimension")
parser.add_argument("--char_emb_dim", dest="char_emb_dim", default=25,
                    help="char embedding dimension")
parser.add_argument("--use_char", dest="use_char", default=True,
                    help="use character embedding?")
parser.add_argument("--use_crf", dest="use_crf", default=True,
                    help="use crf as the final layer?")

from collections import namedtuple
converted_data = namedtuple('ConvertedData', 'tokens labels')

import json
class InputParser(object):
    """
    --> convert input train/validation data to sequence of tokens and sequence of labels
    """
    def __init__(self,
                 path_to_data='',
                 tokenizer=RegexpTokenizer(r'\w+|[^\w\s]')):
        """
        :param path_to_data (json file): path of data to be converted, no validation of path
        :param tokenizer: word tokenizer, inputs string returns sequence of tokens.
                            The tokenizer should have the method span_tokenize, which returns the span of all
                            the tokens relative to the text.
        """
        self._tokenizer = tokenizer
        self._data_path = path_to_data
        self._ent_labels = set()

    def convert(self):
        """
        --> Convert {'text' : 'I have the power.', 'entities' : [11,16,'entity']}
            to (['I', 'have', 'the', 'power', '.'], ['O', 'O', 'O', 'S-entity', 'O'])
        :return: list of named tuples
        """
        with open(self._data_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)

        all_items = []

        for idx, item in enumerate(obj):
            all_items.append(self._convert_single_item(item, idx))

        return all_items

    def _convert_single_item(self, item, item_idx):
        """
        --> convert single item
        --> assumes each token is a part of at most one entity
        :param item: dict , {'text' : 'example', 'entities': [0, 1, 'E']}
        :return: named tuple

        ==> why not to break the text according to entity spans and then find the labels
        Ans. Consider, {'text' : 'I have the power.', 'entities': [9,16, 'ent']},
            1. when breaking according to entity spans the tokens would be
                ['I', 'have', 'th', 'e', 'power', '.'] and the labels would be
                ['O', 'O', 'O', 'B-ent', 'I-ent', 'O']
            2. when not breaking according to the entity spans the tokens would be
                ['I', 'have', 'the', 'power', '.'] and labels would not be created
                since entity covers some part of 'the' and full part of 'power'.
            Therefore, do not break text according to entity spans. For finding labels, match the entity span
            with token spans. An entity might cover multiple full tokens. In case of partial span match throw warning.
        """
        spans = list(self._tokenizer.span_tokenize(item['text']))
        ent_to_tok_maps = [(-1, -1, '')]*len(item['entities'])
        tok_start = [idx for idx, _ in spans]
        tok_end = [idx for _, idx in spans]

        for idx, (start, stop, label) in enumerate(item['entities']):
            start_tok = tok_start.index(start) if start in tok_start else -1
            end_tok = tok_end.index(stop) if stop in tok_end else -1

            if start_tok != -1 and end_tok != -1:
                ent_to_tok_maps[idx] = [start_tok, end_tok, label]
                self._ent_labels.add(label)
            else:
                logger.warning('{} :: entity number {} in item {} could not be converted'.format(self.__class__.__name__,
                                                                                                 idx,
                                                                                                 item_idx))

        non_ent_tokens = set(range(0,
                                   len(spans))).difference(set(reduce(lambda x, y: x+y,
                                                [(start, stop) for start, stop,_ in ent_to_tok_maps])))

        labels = ['']*len(spans)
        for tok_idx in non_ent_tokens:
            labels[tok_idx] = 'O'

        for ent_to_tok_map in ent_to_tok_maps:
            for tok_idx, label in InputParser._ent_to_labels(ent_to_tok_map):
                labels[tok_idx] = label

        return self._tokenizer.tokenize(item['text']), labels

    @staticmethod
    def _ent_to_labels(ent_to_tok_map):
        """
        --> convert "entity to token map" to labels
        --> convert single entity to labels
        :param ent_to_tok_map: list
        :return (labels): list
        """
        start, stop, label = ent_to_tok_map

        if start == stop == -1 and label == '':
            return []

        total_toks = list(range(start, stop + 1))

        if len(total_toks) == 1:
            return [(total_toks[0], 'S-{}'.format(label))]
        elif len(total_toks) == 2:
            return [(total_toks[0], 'B-{}'.format(label)),
                    (total_toks[1], 'E-{}'.format(label))]
        else:
            obj = [(total_toks[0], 'B-{}'.format(label))]
            obj += [(tok_idx, 'I-{}'.format(label)) for tok_idx in total_toks[1:-1]]
            obj += [(total_toks[-1], 'E-{}'.format(label))]
            return obj

    def get_labels(self):
        """
        --> get entity labels
        :return: list
        """
        all_labels = ['{}-{}'.format(st, item) for item in self._ent_labels for st in ['B', 'I', 'E', 'S']] + ['O']
        return all_labels

if __name__ == '__main__':
    args = parser.parse_args()
    path_w2v = Path(args.path_w2v.replace("\\",''))
    word_emb_dim = args.word_emb_dim
    char_emb_dim = args.char_emb_dim
    use_char = args.use_char
    use_crf = args.use_crf

    path_train = Path(args.path_train.replace("\\",''))
    if not path_train.is_file():
        logger.error('{} is not a valid file'.format(path_train))
        exit(1)

    path_valid = Path(args.path_valid.replace("\\",''))
    if not path_valid.is_file():
        logger.error('{} is not a valid file'.format(path_valid))
        exit(1)

    train_parser = InputParser(path_to_data=path_train)
    valid_parser = InputParser(path_to_data=path_valid)

    train_data = train_parser.convert()
    valid_data = valid_parser.convert()

    model_wrapper = ModelWrapper(
        word_embedding_dim=word_emb_dim,
        char_embedding_dim=char_emb_dim,
        use_char=use_char,
        use_crf=use_crf,
        path_to_w2v=path_w2v,
        all_labels=train_parser.get_labels()
    )

    print(train_data)
    model_wrapper.fit([x for x, _ in train_data],
                      [y for _, y in train_data],
                      x_valid=[x for x, _ in valid_data],
                      y_valid=[y for _, y in valid_data])