import tensorflow as tf
import tensorflow.keras.layers as L



class EncoderLSTM(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units,
                 embedding_matrix='uniform',train_emd=True, bid=False):
        
        super(EncoderLSTM, self).__init__()
        #self.batch_sz = batch_sz
        
        self.enc_units = enc_units
        self.is_bid = bid
    
        # embedding layer output shape == (batch_size, seq_length, embedding_size)
        self.embedding = L.Embedding(
            vocab_size,
            embedding_dim,
            embeddings_initializer = embedding_matrix,
            trainable=train_emd)
    
        if self.is_bid:
            self.lstm = L.Bidirectional(L.LSTM(
                self.enc_units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer='glorot_uniform'))

        else:
            self.lstm = L.LSTM(
                self.enc_units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        if self.is_bid:
            output, state, f_c,b_s,b_c = self.lstm(x, initial_state = hidden)
            state = tf.concat([state, b_s],axis=-1)
        else:
            output, state, _ = self.lstm(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self, batch_sz):
        if self.is_bid:
            return [tf.zeros((batch_sz, self.enc_units)) for i in range(4)]
        else:
            return [tf.zeros((batch_sz, self.enc_units)) for i in range(2)]




class EncoderGRU(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units,
                 embedding_matrix='uniform',train_emd=True, bid=False):
        
        super(EncoderGRU, self).__init__()
        #self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.is_bid = bid
        
        # embedding layer output shape == (batch_size, seq_length, embedding_size)
        self.embedding = L.Embedding(
            vocab_size,
            embedding_dim,
            embeddings_initializer=embedding_matrix,
            trainable=train_emd)
        
        if self.is_bid:
            self.gru = L.Bidirectional(L.GRU(
                self.enc_units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer='glorot_uniform'))
        else:
            self.gru = L.GRU(
                self.enc_units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        if self.is_bid:
            output, state,b_s = self.gru(x, initial_state = hidden)
            state = tf.concat([state,b_s],axis=-1)
        else:
            output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self, batch_sz):
        if self.is_bid:
            return [tf.zeros((batch_sz, self.enc_units)) for i in range(2)]
        else:
            return [tf.zeros((batch_sz, self.enc_units)) for i in range(1)]









