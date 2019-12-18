from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, TimeDistributed
from keras.layers.merge import Concatenate
from keras.models import Model
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_NER.__common__ import DTYPE, LOGGER_NAME
import logging
logger = logging.getLogger(LOGGER_NAME)
from copy import deepcopy

class BiLSTMCRF(object):
    """
    --> A Keras implementation of BiLSTM-CRF for sequence labeling.
    """

    def __init__(self,
                max_seq_len=50,
                max_tok_len=20,
                tok_emb_dim=64,
                char_emb_dim=16,
                char_lvl_tok_emb_dim=32,
                char_vocab_size=26,
                tok_vocab_size=1000,
                lstm_size=256,
                use_char=True,
                tok_emb=None,
                dropout=0.3,
                use_crf=True,
                num_labels=12,
                optimizer='adam'
        ):
        """
        --> Build a Bi-LSTM CRF model.

        """
        super(BiLSTMCRF).__init__()
        self._max_seq_len = max_seq_len
        self._max_tok_len = max_tok_len
        self._tok_emb_dim = tok_emb_dim
        self._char_emb_dim = char_emb_dim
        self._char_lvl_tok_emb_dim = char_lvl_tok_emb_dim
        self._char_vocab_size = char_vocab_size
        self._tok_vocab_size = tok_vocab_size
        self._lstm_units = lstm_size
        self._use_char = use_char
        self._tok_emb = tok_emb
        self._dropout = dropout
        self._fully_connected_dim = 512
        self._fully_connected_act = 'tanh'
        self._use_crf = use_crf
        self._num_labels = num_labels
        self._optimizer = optimizer

        self._model = None
        self._loss = None

    def _get_embedding(self):
        """
        --> return embedding layer
        :return: embedding
        """
        tok_ids = Input(shape=(self._max_seq_len,), dtype=DTYPE['int'], name='input_tokens')
        inputs = [tok_ids]
        if self._tok_emb is None:
            tok_emb = Embedding(input_dim=self._tok_vocab_size,
                                        output_dim=self._tok_emb_dim,
                                        mask_zero=True,
                                        name='token_embedding')(tok_ids)
            logger.info('{} :: initializing word embedding with random weights'.format(self.__class__.__name__))
        else:
            tok_emb = Embedding(input_dim=self._tok_emb.shape[0],
                                        output_dim=self._tok_emb.shape[1],
                                        mask_zero=True,
                                        weights=[self._tok_emb],
                                        name='token_embedding')(tok_ids)
            logger.info('{} :: initializing word embedding with the weights provided'.format(self.__class__.__name__))

        # build character based word embedding
        if self._use_char:
            char_ids = Input(shape=(self._max_seq_len, self._max_tok_len,),
                             dtype=DTYPE['int'], name='char_input')
            inputs.append(char_ids)
            char_emb = Embedding(input_dim=self._char_vocab_size,
                                        output_dim=self._char_emb_dim,
                                        mask_zero=True,
                                        name='char_embedding')(char_ids)
            char_emb = TimeDistributed(Bidirectional(LSTM(self._char_emb_dim//2)))(char_emb)
            tok_emb = Concatenate()([tok_emb, char_emb])
            logger.info('{} :: using char embedding'.format(self.__class__.__name__))

        return inputs, tok_emb

    def build(self):
        logger.info('{} :: building the bilstm model'.format(self.__class__.__name__))

        inputs, embeddings = self._get_embedding()

        embeddings = Dropout(self._dropout)(embeddings)
        _temp = Bidirectional(LSTM(units=self._lstm_units, return_sequences=True))(embeddings)
        _temp = TimeDistributed(Dense(units=self._fully_connected_dim,
                                      activation=self._fully_connected_act))(_temp)

        if self._use_crf:
            crf = CRF(self._num_labels, sparse_target=True)
            loss = crf_loss
            pred = crf(_temp)
            logger.info('{} :: using CRF as the final layer'.format(self.__class__.__name__))
        else:
            loss = 'categorical_crossentropy'
            pred = TimeDistributed(Dense(self._num_labels, activation='softmax'))(_temp)
            logger.info('{} :: using fully connected layer with "softmax activation" and "categorical crossentropy" '
                        'loss as final layer'.format(self.__class__.__name__))

        model = Model(inputs=inputs, outputs=pred)

        self._model = model
        self._loss = loss

    def compile(self):
        self._model.compile(loss=self._loss, optimizer=self._optimizer)

    def fit_generator_(self,
                       generator=None,
                       epochs=5,
                       callbacks=None,
                       verbose=0,
                       shuffle=True):
        """
        --> fits generator on the self._model variable
        --> does not stores the fitted model
        --> returns fitted model
        :param generator:
        :param epochs:
        :param callbacks:
        :param verbose:
        :param shuffle:
        :return:
        """
        model = deepcopy(self._model)
        model.fit_generator(
            generator=generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=shuffle
        )

        return model



class BiLSTMCRF_(object):
    """
    --> A Keras implementation of BiLSTM-CRF for sequence labeling.
    """

    def __init__(self,
                 num_labels=None,
                 word_vocab_size=None,
                 char_vocab_size=None,
                 word_embedding_dim=100,
                 char_embedding_dim=25,
                 word_lstm_size=100,
                 char_lstm_size=25,
                 fc_dim=100,
                 fc_act='tanh',
                 optimizer = 'adam',
                 dropout=0.5,
                 embeddings=None,
                 use_char=True,
                 use_crf=True):
        """
        --> Build a Bi-LSTM CRF model.

        :param word_vocab_size (int): word vocabulary size.
        :param char_vocab_size (int): character vocabulary size.
        :param num_labels (int): number of entity labels.
        :param word_embedding_dim (int): word embedding dimensions.
        :param char_embedding_dim (int): character embedding dimensions.
        :param word_lstm_size (int): character LSTM feature extractor output dimensions.
        :param char_lstm_size (int): word tagger LSTM output dimensions.
        :param fc_dim (int): output fully-connected layer size.
        :param fc_act (str): output fully-connected activation
        :param dropout (float): dropout rate.
        :param embeddings (numpy array): word embedding matrix.
        :param use_char (boolean): add char feature.
        :param use_crf (boolean): use crf as last layer.
        """
        super(BiLSTMCRF).__init__()
        self._char_embedding_dim = char_embedding_dim
        self._word_embedding_dim = word_embedding_dim
        self._char_lstm_size = char_lstm_size
        self._word_lstm_size = word_lstm_size
        self._char_vocab_size = char_vocab_size
        self._word_vocab_size = word_vocab_size
        self._fc_dim = fc_dim
        self._fc_act = fc_act
        self._dropout = dropout
        self._use_char = use_char
        self._use_crf = use_crf
        self._embeddings = embeddings
        self._num_labels = num_labels
        self._model = None
        self._loss = None
        self._optimizer = optimizer
,
    def _get_embedding(self):
        """
        --> return embedding layer
        :return: embedding
        """
        word_ids = Input(batch_shape=(None, None), dtype=DTYPE['int'], name='word_input')
        inputs = [word_ids]
        if self._embeddings is None:
            word_embeddings = Embedding(input_dim=self._word_vocab_size,
                                        output_dim=self._word_embedding_dim,
                                        mask_zero=True,
                                        name='word_embedding')(word_ids)
            logger.info('{} :: initializing word embedding with random weights'.format(self.__class__.__name__))
        else:
            word_embeddings = Embedding(input_dim=self._embeddings.shape[0],
                                        output_dim=self._embeddings.shape[1],
                                        mask_zero=True,
                                        weights=[self._embeddings],
                                        name='word_embedding')(word_ids)
            logger.info('{} :: initializing word embedding with the weights provided'.format(self.__class__.__name__))

        # build character based word embedding
        if self._use_char:
            char_ids = Input(batch_shape=(None, None, None), dtype=DTYPE['int'], name='char_input')
            inputs.append(char_ids)
            char_embeddings = Embedding(input_dim=self._char_vocab_size,
                                        output_dim=self._char_embedding_dim,
                                        mask_zero=True,
                                        name='char_embedding')(char_ids)
            char_embeddings = TimeDistributed(Bidirectional(LSTM(self._char_lstm_size)))(char_embeddings)
            word_embeddings = Concatenate()([word_embeddings, char_embeddings])
            logger.info('{} :: using char embedding'.format(self.__class__.__name__))

        return inputs, word_embeddings

    def build(self):
        # build word embedding
        logger.info('{} :: building the bilstm model'.format(self.__class__.__name__))

        inputs, word_embeddings = self._get_embedding()

        word_embeddings = Dropout(self._dropout)(word_embeddings)
        z = Bidirectional(LSTM(units=self._word_lstm_size, return_sequences=True))(word_embeddings)
        z = Dense(self._fc_dim, activation=self._fc_act)(z)

        if self._use_crf:
            crf = CRF(self._num_labels)
            loss = crf.loss_function
            pred = crf(z)
            logger.info('{} :: using CRF as the final layer'.format(self.__class__.__name__))
        else:
            loss = 'categorical_crossentropy'
            pred = Dense(self._num_labels, activation='softmax')(z)
            logger.info('{} :: using fully connected layer with "softmax activation" and "categorical crossentropy" '
                        'loss as final layer'.format(self.__class__.__name__))

        model = Model(inputs=inputs, outputs=pred)

        self._model = model
        self._loss = loss

    def compile(self):
        self._model.compile(loss=self._loss, optimizer=self._optimizer)

    def fit_generator(self,
                      generator=None,
                      epochs=5,
                      callbacks=None,
                      verbose=0,
                      shuffle=True):
        """
        --> fits generator on the self._model variable
        --> does not stores the fitted model
        --> returns fitted model
        :param generator:
        :param epochs:
        :param callbacks:
        :param verbose:
        :param shuffle:
        :return:
        """
        model = deepcopy(self._model)
        model.fit_generator_(
            generator=generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=shuffle
        )

        return model