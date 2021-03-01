import tensorflow as tf
import tensorflow.keras.layers as L
from AttentionLayers import BahdanauAttention
from AttentionLayers import ScaledDotProductAttention


class DecoderLSTM(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units,
               embedding_matrix='uniform',train_emd=True, bid=False, att='bahdanau'):
    
    super(DecoderLSTM, self).__init__()
    #self.batch_sz = batch_sz
    self.is_bid = bid
    self.att = att
    
    if self.is_bid:
        self.dec_units = dec_units*2
    else:
        self.dec_units = dec_units
        
    self.embedding = L.Embedding(
            vocab_size,
            embedding_dim,
            embeddings_initializer=embedding_matrix,
            trainable=train_emd)

    self.lstm = L.LSTM(
        self.dec_units,
        return_sequences = True,
        return_state = True,
        recurrent_initializer = 'glorot_uniform')    
    
    self.fc = L.Dense(vocab_size, name = 'Decoder_dense')

    # used for attention
    
    if self.att is 'bahdanau':
        self.attention = BahdanauAttention(self.dec_units)
    elif self.att is 'sdpa':
        self.attention = ScaledDotProductAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # x: is the input word
    # hidden: is the previous hidden state of decoder RNN
    # enc_output: are all the hidden states of input sequence
    
    if self.att is 'bahdanau':
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)
    elif self.att is 'sdpa':
        context_vector, attention_weights = self.attention(hidden, enc_output,  enc_output)
    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state, _ = self.lstm(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights


class DecoderGRU(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units,
               embedding_matrix='uniform',train_emd=True, bid=False, att='bahdanau'):
    
    super(DecoderGRU, self).__init__()
    #self.batch_sz = batch_sz
    self.is_bid = bid
    self.att = att
    
    if self.is_bid:
        self.dec_units = dec_units*2
    else:
        self.dec_units = dec_units
    
    self.embedding = L.Embedding(
            vocab_size,
            embedding_dim,
            embeddings_initializer=embedding_matrix,
            trainable=train_emd)

    self.gru = L.GRU(
        self.dec_units,
        return_sequences = True,
        return_state = True,
        recurrent_initializer = 'glorot_uniform')
    
    self.fc = L.Dense(vocab_size, name = 'Decoder_dense')
    
    if self.att is 'bahdanau':
        self.attention = BahdanauAttention(self.dec_units)
    elif self.att is 'sdpa':
        self.attention = ScaledDotProductAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    
    if self.att is 'bahdanau':
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)
    elif self.att is 'sdpa':
        context_vector, attention_weights = self.attention(hidden, enc_output,  enc_output)
        
    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    output, state = self.gru(x)
    output = tf.reshape(output, (-1, output.shape[2]))
    x = self.fc(output)

    return x, state, attention_weights
