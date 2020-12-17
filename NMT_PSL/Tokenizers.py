import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
from transformers import AutoTokenizer

def TFTokenizer(
    train_data,test_data, inp_lang, targ_lang, 
    pad='post', fit_test=True, is_lower=True,
    max_len_inp=None, max_len_targ=None, verbose=True):
    
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

    if verbose:
        print('Max_length of input sequence train: {}'.format(inp_tensor_train.shape[1]))
        print('Max_length of target sequence train: {}'.format(targ_tensor_train.shape[1]))
        print('Max_length of input sequence test: {}'.format(inp_tensor_test.shape[1]))
        print('Max_length of target sequence test: {}'.format(targ_tensor_test.shape[1]))
    
    return tokenizer, (inp_tensor_train, targ_tensor_train, inp_tensor_test, targ_tensor_test)


def TFSubTokenizer(
    train_data, test_data, inp_lang, targ_lang, pad='post',
    fit_test=True, is_lower=True, verbose=True):

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

    if verbose:
        print('Max_length of input sequence train: {}'.format(inp_tensor_train.shape[1]))
        print('Max_length of target sequence train: {}'.format(targ_tensor_train.shape[1]))
        print('Max_length of input sequence test: {}'.format(inp_tensor_test.shape[1]))
        print('Max_length of target sequence test: {}'.format(targ_tensor_test.shape[1]))
    
    return tokenizer, (inp_tensor_train, targ_tensor_train, inp_tensor_test, targ_tensor_test)


def bertTokenizer(train_data,test_data, inp_lang, targ_lang, pad='post'):
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    train_inp = np.array(train_data[inp_lang])
    train_targ = np.array(train_data[targ_lang])
    test_inp = np.array(test_data[inp_lang])
    test_targ = np.array(test_data[targ_lang])
    
    #max_length_inp = train_data[inp_lang].map(lambda x: len(x.split(' '))).max()
    #max_length_targ = train_data[targ_lang].map(lambda x: len(x.split(' '))).max()
    #print('Max_length of input sequence: {}'.format(max_length_inp))
    #print('Max_length of target sequence: {}'.format(max_length_targ))
    
    max_length_inp = None
    max_length_targ = None
    padd = True
    
    train_inp_encoded = tokenizer(
        list(train_inp),
        max_length= max_length_inp,
        padding= padd, 
        return_tensors= 'tf', truncation=True)['input_ids']

    train_targ_encoded = tokenizer(
        list(train_targ),
        max_length= max_length_targ,
        padding= padd, 
        return_tensors= 'tf', truncation=True)['input_ids']

    test_inp_encoded = tokenizer(
        list(test_inp),
        max_length= max_length_inp,
        padding= padd, 
        return_tensors= 'tf', truncation=True)['input_ids']

    test_targ_encoded = tokenizer(
        list(test_targ),
        max_length= max_length_targ,
        padding= padd, 
        return_tensors= 'tf', truncation=True)['input_ids']
    
    #max_length_inp = max(list(map(lambda x: len(x), train_inp_encoded)))
    #max_length_targ = max(list(map(lambda x: len(x), train_targ_encoded)))
    #max_length_inp_test = max(list(map(lambda x: len(x), test_inp_encoded)))
    #max_length_targ_test = max(list(map(lambda x: len(x), test_targ_encoded)))

    #print('Max_length of input sequence train: {}'.format(max_length_inp))
    #print('Max_length of target sequence train: {}'.format(max_length_targ))
    #print('Max_length of input sequence test: {}'.format(max_length_inp_test))
    #print('Max_length of target sequence test: {}'.format(max_length_targ_test))
    
    return tokenizer,(train_inp_encoded, 
                      train_targ_encoded, 
                      test_inp_encoded, 
                      test_targ_encoded
                     )