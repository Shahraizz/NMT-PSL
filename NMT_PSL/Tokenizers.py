import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def TFTokenizer(train_data,test_data, inp_lang, targ_lang, pad='post'):
    
    train_inp = np.array(train_data[inp_lang])
    train_targ = np.array(train_data[targ_lang])
    test_inp = np.array(test_data[inp_lang])
    test_targ = np.array(test_data[targ_lang])
    
    max_length_inp = train_data[inp_lang].map(lambda x: len(x.split(' '))).max()
    max_length_targ = train_data[targ_lang].map(lambda x: len(x.split(' '))).max()
    print('Max_length of input sequence: {}'.format(max_length_inp))
    print('Max_length of target sequence: {}'.format(max_length_targ))
    
    tokenizer = Tokenizer(filters='',oov_token='<oov>')
    tokenizer.fit_on_texts(train_inp)
    tokenizer.fit_on_texts(train_targ)
    tokenizer.fit_on_texts(test_inp)
    tokenizer.fit_on_texts(test_targ)
    
    inp_tensor_train = pad_sequences(
        tokenizer.texts_to_sequences(train_inp), padding=pad, maxlen=max_length_inp)
    targ_tensor_train = pad_sequences(
        tokenizer.texts_to_sequences(train_targ), padding=pad, maxlen=max_length_targ)
    inp_tensor_test = pad_sequences(
        tokenizer.texts_to_sequences(test_inp), padding=pad, maxlen=max_length_inp)
    targ_tensor_test = pad_sequences(
        tokenizer.texts_to_sequences(test_targ), padding=pad, maxlen=max_length_targ)
    
    return tokenizer, (inp_tensor_train, targ_tensor_train, inp_tensor_test, targ_tensor_test)
