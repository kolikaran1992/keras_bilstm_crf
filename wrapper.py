from keras_NER.model import BiLSTMCRF
from keras_NER.sequence_encoder import SequenceEncoder
from seqeval.metrics import f1_score
from keras_NER.__utils__ import save_model, load_model
from keras_NER.__common__ import LOGGER_NAME
import logging
logger = logging.getLogger(LOGGER_NAME)
from keras_NER.callbacks import F1score
from keras_NER.sequence_encoder import NERSequence
from keras_NER.__paths__ import path_obj
from pickle import dump, load

class ModelWrapper(object):
    def __init__(self,
                 word_embedding_dim=100,
                 char_embedding_dim=25,
                 word_lstm_size=100,
                 char_lstm_size=25,
                 fc_dim=100,
                 dropout=0.5,
                 use_char=True,
                 use_crf=True,
                 path_to_w2v='',
                 all_labels=('O',),
                 optimizer='adam',
                 max_seq_len=50,
                 max_word_len=10):

        self.model = None
        self._seq_enc = None
        self.tagger = None

        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.word_lstm_size = word_lstm_size
        self.char_lstm_size = char_lstm_size
        self.fc_dim = fc_dim
        self.dropout = dropout
        self.use_char = use_char
        self.use_crf = use_crf
        self._path_to_w2v = path_to_w2v
        self._all_labels = all_labels
        self.optimizer = optimizer
        self._max_seq_len = max_seq_len
        self._max_word_len = max_word_len

    def fit(self, x_train, y_train, x_valid=None, y_valid=None,
            epochs=1, batch_size=32, verbose=1, callbacks=None, shuffle=True):
        """
        --> Fit the model for a fixed number of epochs.

        :param x_train: list of training data.
        :param y_train: list of training target (label) data.
        :param x_valid: list of validation data.
        :param y_valid: list of validation target (label) data.
        :param batch_size: Integer.
            Number of samples per gradient update.
            If unspecified, `batch_size` will default to 32.
        :param epochs: Integer. Number of epochs to train the model.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param callbacks: List of `keras.callbacks.Callback` instances.
            List of callbacks to apply during training.
        :param shuffle: Boolean (whether to shuffle the training data
            before each epoch). `shuffle` will default to True.
        """
        seq_enc = SequenceEncoder(path_to_w2v=self._path_to_w2v,
                                  label_vocab=self._all_labels,
                                  char=self.use_char,
                                  max_seq_len=self._max_seq_len,
                                  max_word_len=self._max_word_len)
        seq_enc.encode_multiple(x_train, all_labels=y_train)
        embeddings = seq_enc.get_word_embedding()

        bilstm = BiLSTMCRF(
            max_seq_len=self._max_seq_len,
            max_tok_len=self._max_word_len,
            tok_emb_dim=seq_enc.get_word_embedding().shape[1],
            char_emb_dim=15,
            char_lvl_tok_emb_dim=self.char_embedding_dim,
            char_vocab_size=seq_enc.get_char_vocab_size(),
            tok_vocab_size=seq_enc.get_word_vocab_size(),
            lstm_size=self.word_lstm_size,
            use_char=self.use_char,
            tok_emb=embeddings,
            dropout=self.dropout,
            use_crf=self.use_crf,
            num_labels=seq_enc.get_label_vocab_size(),
            optimizer=self.optimizer
            )

        bilstm.build()
        bilstm.compile()

        train_seq = NERSequence(x_train, y_train, batch_size)

        if x_valid and y_valid:
            valid_seq = NERSequence(x_valid, y_valid, batch_size)
            f1 = F1score(valid_seq)
            callbacks = [f1] + callbacks if callbacks else [f1]

        self._model = bilstm.fit_generator_(generator=train_seq,
                                            epochs=epochs,
                                            callbacks=callbacks,
                                            verbose=verbose,
                                            shuffle=shuffle)
        self._seq_enc = seq_enc


    def predict(self, x_test):
        """Returns the prediction of the model on the given test data.
        Args:
            x_test : array-like, shape = (n_samples, sent_length)
            Test samples.
        Returns:
            y_pred : array-like, shape = (n_smaples, sent_length)
            Prediction labels for x.
        """
        if self.model:
            lengths = map(len, x_test)
            x_test = self._seq_enc.encode_multiple(x_test)
            y_pred = self.model.predict(x_test)
            y_pred = self._seq_enc.decode(y_pred, lengths)
            return y_pred
        else:
            raise OSError('Could not find a model. Call load(dir_path).')

    def score(self, x_test, y_test):
        """Returns the f1-micro score on the given test data and labels.
        Args:
            x_test : array-like, shape = (n_samples, sent_length)
            Test samples.
            y_test : array-like, shape = (n_samples, sent_length)
            True labels for x.
        Returns:
            score : float, f1-micro score.
        """
        if self.model:
            x_test = self.p.transform(x_test)
            lengths = map(len, y_test)
            y_pred = self.model.predict(x_test)
            y_pred = self.p.inverse_transform(y_pred, lengths)
            score = f1_score(y_test, y_pred)
            return score
        else:
            raise OSError('Could not find a model. Call load(dir_path).')

    def analyze(self, text, tokenizer=str.split):
        """Analyze text and return pretty format.
        Args:
            text: string, the input text.
            tokenizer: Tokenize input sentence. Default tokenizer is `str.split`.
        Returns:
            res: dict.
        """
        if not self.tagger:
            self.tagger = Tagger(self.model,
                                 preprocessor=self.p,
                                 tokenizer=tokenizer)

        return self.tagger.analyze(text)

    def save(self, model_name):
        weights_path = path_obj.joinpath('Saved Objects', 'Model Weights')
        if not weights_path.is_dir():
            weights_path.mkdir(parents=True, exist_ok=True)
        weights_path.joinpath(model_name)

        params_path = path_obj.joinpath('Saved Objects', 'Model Parameters')
        if not params_path.is_dir():
            params_path.mkdir(parents=True, exist_ok=True)
        params_path.joinpath(model_name)

        self._seq_enc.save(model_name)
        save_model(self.model, weights_path.as_posix(), params_path.as_posix())

        wrapper_path = path_obj.joinpath('Saved Objects', 'Wrapper', model_name)
        if not wrapper_path.is_dir():
            wrapper_path.mkdir(parents=True, exist_ok=True)
        for att_name, att_val in vars(self).items():
            if att_name in ['model', '_seq_enc']:
                continue
            dump(att_val, wrapper_path.joinpath(att_name))

    @classmethod
    def load(cls, model_name):
        self = cls()

        wrapper_path = path_obj.joinpath('Saved Objects', 'Wrapper', model_name)
        if not wrapper_path.is_dir():
            logger.error('{} does not exists'.format(wrapper_path))
        for file_path in wrapper_path.glob('*'):
            file_name = file_path.name
            obj = load(file_path)
            setattr(self, file_name, obj)

        weights_path = path_obj.joinpath('Saved Objects', 'Model Weights', model_name)
        if not weights_path.is_file():
            logger.error('{} does not exists'.format(weights_path))
        weights_path.joinpath(model_name)

        params_path = path_obj.joinpath('Saved Objects', 'Model Parameters')
        if not params_path.is_dir():
            logger.error('{} does not exists'.format(params_path))
        params_path.joinpath(model_name)

        seq_enc = SequenceEncoder()
        seq_enc.load(model_name)
        self.seq_enc = seq_enc
        self.model = load_model(weights_path.as_posix(), params_path.as_posix())

        return self