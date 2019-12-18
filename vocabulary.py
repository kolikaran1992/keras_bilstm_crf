from keras_NER.__common__ import LOGGER_NAME
import logging

logger = logging.getLogger(LOGGER_NAME)


class Vocabulary(object):
    """
    --> map/reverse_map tokens to ints
    """

    def __init__(self,
                 vocab=(),
                 specials=('<unk>', '<pad>')):
        """
        :param vocab: tuple of all the tokens to be included in the vocabulary
        :param specials: tuple of special tokens that will be prepended to the vocabulary.
        """
        self._specials = list(specials)
        self._token2id = {token: i for i, token in enumerate(self._specials + vocab)}
        self._id2token = list(self._specials + vocab)

    def __len__(self):
        return len(self._token2id)

    def doc2id(self, tokens):
        """
        --> Get the list of token_id given doc.
        :param tokens (list): a list of tokens.
        :return list: int id of doc.
        """
        return [self.token_to_id(token) for token in tokens]

    def id2doc(self, ids):
        """
        --> Get the token list.
        :param ids (list): token ids.
        :return list: token list.
        """
        return [self.id_to_token(idx) for idx in ids]

    def token_to_id(self, token):
        """
        --> Get the token_id of given token.
        :param token (str): token from vocabulary.
        :return int: int id of token.
        """
        return self._token2id.get(token, self._token2id['<unk>'])

    def id_to_token(self, idx):
        """
        --> token-id to token (string).
        :param idx (int): token id.
        :return str: string of given token id.
        """
        return self._id2token[idx]

    def pad_sequence(self,
                     tokens,
                     length):
        """
        --> pads the input sequence to given length
        :param tokens (list): sequence of tokens
        :param length (int): length to be padded
        :return list: padded sequence
        """
        return tokens + ['<pad>'] * (length - len(tokens))

    @property
    def vocab(self):
        """
        --> Return the vocabulary.
        :return dict: get the dict object of the vocabulary.
        """
        return self._token2id

    @property
    def reverse_vocab(self):
        """
        --> Return the vocabulary as a reversed dict object.
        :return dict: reversed vocabulary object.
        """
        return self._id2token
