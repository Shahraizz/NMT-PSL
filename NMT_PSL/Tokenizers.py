import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
from transformers import AutoTokenizer
from bpemb import BPEmb

from nltk.tokenize import TweetTokenizer


def preprocess_sentence(w):
    tknzr = TweetTokenizer()
    x = tknzr.tokenize(w)
    x = [i.lower() for i in x]
    return ' '.join(x)

def start_end_token(sen):
    return "<start> "+sen+" <end>"

def TFTokenizer(
    train_data,test_data, inp_lang, targ_lang, 
    pad='post', fit_test=True, is_lower=True,
    max_len_inp=None, max_len_targ=None, verbose=True):
    
    """
    Args:

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
    Args:
    train_data: An object of pandas.DataFrame containing training data.
    test_data: An object of pandas.DataFrame containing test data.
    inp_lang: Input language name.
    targ_lang: Target language name.

    Return:
    
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




def bpeTokenizer(train_data, test_data, inp_lang, targ_lang, vs, pad='post',
                 fit_test=True, is_lower=True, verbose=True):
    
    """
    Args:
    train_data: An object of pandas.DataFrame containing training data.
    test_data: An object of pandas.DataFrame containing test data.
    inp_lang: Input language name.
    targ_lang: Target language name.
    vs: Vocabulary size (1000, 3000, 10000, 50000, 100000)

    Return:
    
    """
    
    bpemb_en = BPEmb(lang="en", vs=vs)
    train_inp = bpemb_en.encode_ids(train_data[inp_lang].values)
    train_targ = bpemb_en.encode_ids(train_data[targ_lang].values)
    test_inp = bpemb_en.encode_ids(test_data[inp_lang].values)
    test_targ = bpemb_en.encode_ids(test_data[targ_lang].values)
    
    ## Adding start and end tokens
    train_inp = [[vs-2]+x+[vs-1] for x in train_inp]
    train_targ = [[vs-2]+x+[vs-1] for x in train_targ]
    test_inp = [[vs-2]+x+[vs-1] for x in test_inp]
    test_targ = [[vs-2]+x+[vs-1] for x in test_targ]
    
    inp_tensor_train = pad_sequences(train_inp, padding=pad)
    targ_tensor_train = pad_sequences(train_targ, padding=pad)
    inp_tensor_test = pad_sequences(test_inp, padding=pad)
    targ_tensor_test = pad_sequences(test_targ, padding=pad)
    
    if verbose:
        print('Max_length of input sequence train: {}'.format(inp_tensor_train.shape[1]))
        print('Max_length of target sequence train: {}'.format(targ_tensor_train.shape[1]))
        print('Max_length of input sequence test: {}'.format(inp_tensor_test.shape[1]))
        print('Max_length of target sequence test: {}'.format(targ_tensor_test.shape[1]))
        
    return bpemb_en, (inp_tensor_train, targ_tensor_train, inp_tensor_test, targ_tensor_test)



def psl_tokenizer(TOKENIZER, df_train, df_test, input_language, targ_language, pad="post"):
    if TOKENIZER == 'tf':
        df_train = df_train.applymap(lambda x: preprocess_sentence(str(x)))
        df_train = df_train.applymap(lambda x: start_end_token(str(x)))
        df_test = df_test.applymap(lambda x: preprocess_sentence(str(x)))
        df_test = df_test.applymap(lambda x: start_end_token(str(x)))
    
        lang, (inp_tensor_train, targ_tensor_train, inp_tensor_test, targ_tensor_test) = TFTokenizer(
            df_train,df_test, input_language,targ_language, pad=pad)
    
        start_tok = lang.word_index['<start>']
        end_tok = lang.word_index['<end>']
    
        vocab_size = len(lang.word_index)+1
    
    elif TOKENIZER is 'tfsub':
        lang, (inp_tensor_train, targ_tensor_train, inp_tensor_test, targ_tensor_test) = TFSubTokenizer(
            df_train, df_test, input_language, targ_language, pad=pad)
    
        start_tok = lang.vocab_size
        end_tok = lang.vocab_size+1
    
        vocab_size = lang.vocab_size
    

    elif TOKENIZER == 'bert':
    
    
        lang, (inp_tensor_train, targ_tensor_train, inp_tensor_test, targ_tensor_test) = bertTokenizer(
            df_train,df_test, input_language,targ_language, pad=pad)
    
        start_tok = lang.cls_token_id
        end_tok = lang.sep_token_id
    
        vocab_size = lang.vocab_size
    

    elif TOKENIZER is 'bpe':
    
        vocab_size = 10000
        start_tok = vocab_size-2
        end_tok = vocab_size-1
    
        lang, (inp_tensor_train, targ_tensor_train, inp_tensor_test, targ_tensor_test) = bpeTokenizer(
            df_train, df_test, input_language, targ_language, vocab_size)

    ######

    max_length_inp = inp_tensor_train.shape[1]
    max_length_targ = targ_tensor_train.shape[1]
    print('Vocabulary size: {}\n'.format(vocab_size))
    ######

    return lang, vocab_size, start_tok, end_tok, inp_tensor_train, targ_tensor_train, inp_tensor_test, targ_tensor_test