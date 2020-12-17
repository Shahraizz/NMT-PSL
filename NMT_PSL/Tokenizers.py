import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds

def TFTokenizer(
    train_data,test_data, inp_lang, targ_lang, 
    pad='post', fit_test=True, is_lower=True, max_len_inp=None, max_len_targ=None):
    
    """
    prameters:

    train_data: An object of pandas.DataFrame containing training data.
    test_data: An object of pandas.DataFrame containing test data.
    inp_lang: Input language name.
    targ_lang: Target language name.
    """

    train_inp = np.array(train_data[inp_lang])
    train_targ = np.array(train_data[targ_lang])
    test_inp = np.array(test_data[inp_lang])
    test_targ = np.array(test_data[targ_lang])
    
    tokenizer = Tokenizer(filters='',oov_token='<oov>', lower=is_lower)
    tokenizer.fit_on_texts(train_inp)
    tokenizer.fit_on_texts(train_targ)

    if fit_test:
        tokenizer.fit_on_texts(test_inp)
        tokenizer.fit_on_texts(test_targ)
    
    inp_tensor_train = pad_sequences(
        tokenizer.texts_to_sequences(train_inp), padding=pad, maxlen=max_len_inp)
    targ_tensor_train = pad_sequences(
        tokenizer.texts_to_sequences(train_targ), padding=pad, maxlen=max_len_targ)
    inp_tensor_test = pad_sequences(
        tokenizer.texts_to_sequences(test_inp), padding=pad, maxlen=max_len_inp)
    targ_tensor_test = pad_sequences(
        tokenizer.texts_to_sequences(test_targ), padding=pad, maxlen=max_len_targ)
    
    return tokenizer, (inp_tensor_train, targ_tensor_train, inp_tensor_test, targ_tensor_test)


def TFSubTokenizer(
    train_data, test_data, inp_lang, targ_lang, pad='post', fit_test=True, is_lower=True):

    """
    prameters:

    train_data: An object of pandas.DataFrame containing training data.
    test_data: An object of pandas.DataFrame containing test data.
    inp_lang: Input language name.
    targ_lang: Target language name.

    """

    if is_lower:
        train_data = train_data.applymap(lambda x: x.lower())
        test_data = test_data.applymap(lambda x: x.lower())
    
    data = list(train_data[inp_lang].values) + list(train_data[targ_lang].values)
    if fit_test:
        data += list(test_data[inp_lang].values)
        data += list(test_data[targ_lang].values)

    corpus_gen = (n for n in data)
    
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(corpus_gen, target_vocab_size=2**13)
    
    def encode(lang):
        lang = [tokenizer.vocab_size] + tokenizer.encode(
            lang) + [tokenizer.vocab_size+1]
        return lang
    
    # Encoding
    train_inp = train_data[inp_lang].apply(encode).values
    train_targ = train_data[targ_lang].apply(encode).values

    test_inp = test_data[inp_lang].apply(encode).values
    test_targ = test_data[targ_lang].apply(encode).values
    
    # Padding sequences
    inp_tensor_train = pad_sequences(train_inp, padding=pad)
    targ_tensor_train = pad_sequences(train_targ, padding=pad)

    inp_tensor_test = pad_sequences(test_inp, padding=pad)
    targ_tensor_test = pad_sequences(test_targ, padding=pad)
    
    return tokenizer, (inp_tensor_train, targ_tensor_train, inp_tensor_test, targ_tensor_test)