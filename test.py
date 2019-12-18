from keras_NER.trainer import InputParser
import json
path = '/home/aptara/PROJECTS/Agel/notebooks/RNN experiments/tensorflow_NER/validation_data.json'
with open(path, 'r', encoding='utf-8') as f:
    obj = json.load(f)

print(obj[582])

#
# new_obj = []
#
# for item in obj:
#     new_obj.append({'text':item[0], 'entities' : item[1]['entities']})
#
# with open('test.json', 'w', encoding='utf-8') as f:
#     json.dump(new_obj, f)

with open('test.json', 'r', encoding='utf-8') as f:
    obj = json.load(f)

print(obj[582])
print(obj[582]['text'][53:61])

inp_parse = InputParser(path_to_data='test.json')
a = inp_parse.convert()
print(a)
print(inp_parse.get_labels())

# from keras_NER.model import BiLSTMCRF
#
# batch_size = 10
# max_seq_len = 50
# max_word_len = 12
# word_emb_dim = 60
# word_vocab_size = 100
# char_emb_dim = 15
#
# char_lvl_word_emb_dim = 32
#
# char_vocab_size = 20
# dtype = 'int64'
#
# char_lstm_size = 11
# lstm_size = 256
#
#
# model = BiLSTMCRF(
#     max_seq_len=max_seq_len,
#     max_tok_len=max_word_len,
#     tok_emb_dim=word_emb_dim,
#     char_emb_dim=char_emb_dim,
#     char_lvl_tok_emb_dim=char_lvl_word_emb_dim,
#     char_vocab_size=char_vocab_size,
#     tok_vocab_size=word_vocab_size,
#     lstm_size=lstm_size,
#     use_char=False,
#     tok_emb=None,
#     dropout=0.3,
#     use_crf=True,
#     num_labels=12,
#     optimizer='adam'
# )
#
# inp, emb = model._get_embedding()
#
# print(inp)
# print(emb)
#
# model.build()
# model.compile()