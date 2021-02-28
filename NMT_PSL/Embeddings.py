import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors



def loadGlove(emd_path, embedding_dim, num_tokens, tokenizer, tok_size):
    
    """
    Args
    
    emd_path: Path of pretrained glove embedding files
    embedding_dim: vector size of each word in the embedding matrix in case of glove select from (50, 100, 200)
    num_tokens: Vocabulary size of given data
    tokenizer: Tokenizer containing the mapping between words to ids and vice versa of training data
    tok_size: Total number of tokens in pre-trained embeddings
    """
    
    if emd_path.get(str(tok_size)+'_'+str(embedding_dim)+'D'):
        path_to_glove_file = emd_path[str(tok_size)+'_'+str(embedding_dim)+'D']
    else:
        print("Can't load glove embeddings")
        return 0   
    
    embeddings_index = {}
    hits = 0
    misses = []

    with open(path_to_glove_file, encoding="utf8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print("Found %s word vectors." % len(embeddings_index))



    # Prepare embedding matrix
    embedding_matrix = np.random.normal(size=(num_tokens, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses.append(word)
    print("Converted %d words (%d misses)" % (hits, len(misses)))
    
    return tf.keras.initializers.Constant(embedding_matrix), misses


def loadFasttext(emd_path, limit, tokenizer, num_tokens, subword=False):
    
    """
    Args
    
    num_tokens: Vocab size of given data
    limit: Top words in Fasttest embeddings
    """
    
    if subword:
        path_to_glove_file = emd_path['subword']
    else:
        path_to_glove_file = emd_path['word']
    
    model = KeyedVectors.load_word2vec_format(path_to_glove_file, limit=limit)
    
    
    hits = 0
    misses = []
    embedding_matrix = np.random.normal(size=(num_tokens, 300))
    for word, i in tokenizer.word_index.items():
        try:
            embedding_vector = model[word]
            embedding_matrix[i] = embedding_vector
            hits += 1
        except KeyError:
            misses.append(word)
            
    print("Converted %d words (%d misses)" % (hits, len(misses)))
    
    
    return tf.keras.initializers.Constant(embedding_matrix), misses


def loadBpeEmb(tokenizer):
    """
    Args
    
    tokenizer: #
    """
    return tf.keras.initializers.Constant(tokenizer.vectors)