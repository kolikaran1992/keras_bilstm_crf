from keras_NER.vocabulary import Vocabulary
from keras_NER.__common__ import LOGGER_NAME
import logging
import math
logger = logging.getLogger(LOGGER_NAME)
from gensim.models import KeyedVectors
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.utils import Sequence
from pickle import load, dump
from keras_NER.__paths__ import path_obj

from collections import namedtuple
encoded_seq = namedtuple('EncodedSequence', 'tok_ids char_ids lab_ids')
decoded_seq = namedtuple('DecodedSequence', 'tokens labels')

class SequenceEncoder(object):
    """
    --> encode input text sequence to a sequence of numbers
    """

    def __init__(self,
                 path_to_w2v='',
                 label_vocab=('O',),
                 char=False,
                 max_seq_len=50,
                 max_word_len=10):
        """
        :param path_to_w2v: path to vectors (word2vec format)
        :param path_to_label_vocab: path to labels file
        :param char: whether to use characters
        :param max_seq_len: maximum sequence length
        :param max_word_len: maximum word length
        """
        self._max_seq_len = max_seq_len
        self._max_word_len = max_word_len

        _temp = KeyedVectors.load_word2vec_format(path_to_w2v)

        self._word_vocab = Vocabulary(vocab=list(_temp.wv.vocab.keys()))
        self._word_vectors = _temp.wv.vectors
        logger.info('{} :: word vocabulary size = {}'.format(self.__class__.__name__, len(self._word_vocab)))

        labels = label_vocab
        self._label_to_1hot = OneHotEncoder()
        self._label_to_1hot.fit(np.array([labels + ['<pad>']]).reshape(-1,1))
        logger.info('{} :: label vocabulary size = {}'.format(self.__class__.__name__, len(labels)+1))

        self._use_char = char
        if char:
            logger.info('{} :: using character encoding'.format(self.__class__.__name__))
            all_chars = list(set([ch for tok in _temp.wv.vocab.keys() for ch in tok]))
            self._char_vocab = Vocabulary(vocab=all_chars)
            logger.info('{} :: char vocabulary size = {}'.format(self.__class__.__name__, len(self._char_vocab)))

        del _temp

    def encode(self, tokens, labels = None):
        """
        --> encoding tokens to ids
        --> pad to maximum len
        :param tokens: list
        :param labels: list
        :return: named tuple
        """
        tokens = self._word_vocab.pad_sequence(tokens, self._max_word_len)
        tok2ids = self._word_vocab.doc2id(tokens)

        char2ids = None
        lab2ids = None

        if labels:
            lab2ids = self._label_to_1hot.transform(np.array([labels]).reshape(-1, 1)).todense().astype(int).tolist()
            lab2ids.extend([self._label_to_1hot.transform([['<pad>']]).todense().astype(int).tolist()[0]] * (self._max_seq_len - len(lab2ids)))
        if self._use_char:
            char2ids = [self._char_vocab.doc2id(self._char_vocab.pad_sequence(list(tok), self._max_word_len)) for tok in tokens] + [
                self._char_vocab.doc2id(self._char_vocab.pad_sequence(['<pad>'], self._max_word_len))] * (self._max_seq_len - len(tokens))

        return encoded_seq(tok_ids=tok2ids, char_ids=char2ids, lab_ids=lab2ids)

    def decode(self, enc_seq):
        """
        --> named tuple containing token ids, char ids, label ids
        --> decode token ids and label ids
        --> remove pad characters
        :param enc_seq: named tuple
        :return: named tuple
        """
        tokens = self._word_vocab.id2doc(enc_seq.tok_ids)
        labels = self._label_to_1hot.inverse_transform(enc_seq.lab_ids).reshape(1,-1).tolist()[0]

        return decoded_seq(tokens=list(filter(lambda x: x != '<pad>', tokens)),
                           labels=list(filter(lambda x: x != '<pad>', labels)))

    def encode_multiple(self, all_tokens, all_labels=None, batch_size=32):
        """
        --> encode list of tokens
        :param all_tokens: list of list
        :param all_labels: list of list
        :return: generator
        """
        if not all_labels:
            all_labels = [None]*len(all_tokens)

        ret = []

        for tokens, labels in zip(all_tokens, all_labels):
            ret.append(self.encode(tokens, labels=labels))

        ret.extend(ret[:len(ret)%batch_size])

        count = 0
        items = []
        for item in ret:
            items.append(item)
            count += 1
            if count % batch_size == 0:
                count = 0
                yield items
                items = []

    def get_word_embedding(self):
        return self._word_vectors

    def get_word_vocab_size(self):
        return len(self._word_vocab)

    def get_char_vocab_size(self):
        return len(self._char_vocab)

    def get_label_vocab_size(self):
        return self._label_to_1hot.n_values

    def save(self, name):
        path = path_obj.joinpath('Saved Objects', 'Sequence Encoders', name)

        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
        for att_name, att_val in vars(self).items():
            dump(att_val, path.joinpath(att_name))

        logger.info('Sequence Encoder saved at {}'.format(path.as_posix()))

    def load(self, name):
        path = path_obj.joinpath('Saved Objects', 'Sequence Encoders', name)

        if not path.is_dir():
            logger.error('{} does not exists'.format(name))

        for file_path in path.glob('*'):
            file_name = file_path.name
            obj = load(file_path)
            setattr(self, file_name, obj)

        logger.info('Sequence Encoder {} loaded successfully')



class NERSequence(Sequence):
    def __init__(self, x, y, batch_size=1, preprocess=lambda *items: items):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.preprocess = preprocess

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]

        return self.preprocess(batch_x, batch_y)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)