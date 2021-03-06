from NMT_PSL.TransformerModel import Transformer
from NMT_PSL.TransformerModel import create_padding_mask
from NMT_PSL.TransformerModel import create_look_ahead_mask
from NMT_PSL.TransformerModel import create_masks
from NMT_PSL import Embeddings
from NMT_PSL import utils
from NMT_PSL.Tokenizers import psl_tokenizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from configparser import ConfigParser
import numpy as np
import pandas as pd
import time


config_file = 'config.ini'
config = ConfigParser()
config.read(config_file)



CONFIG = { 
    "preprocessing":{
        "tokenizer": 'bpe',
        "input_lanuage": "English",
        "targ_language": "PSL",
        "val_rand_state": 37,
        "val_size": 3000,
        "batch_size_train": 64,
        "batch_size_test": 500,
        "batch_size_dev": 1000,
    },
    "transformer":{
        "num_layers": 6,
        "dff": 2048,
        "num_heads": 4,
        "bert_enc": False,
        "d_model": 300,
        "tok_size": 42,
        "dropout_rate": 0.1,
        "embedding": 'bpe',
        "train_emb": False
    }
}

TOKENIZER = 'bpe' ## [tf, tfsub, bpe, bert]
input_language = "English"
targ_language = "PSL"

random_state = 37
val_size = 3000

BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 500
BATCH_SIZE_VAL = 500

EPOCHS = 5

# transformer base parameters
num_layers = 4
dff = 2048
num_heads = 4
bert_enc = False

d_model = 300
tok_size = 42
train_emb = False

embedding = 'bpe'
dropout_rate = 0.1




loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='val_accuracy')




def inference_step(inp, targ, start_tok, transformer, bert_enc=False):
    
    max_len = targ.shape[1]
    encoder_input = inp
    targ_real = targ[:,1:]
    
    decoder_input = [start_tok] * inp.shape[0]
    output = np.reshape(decoder_input,(inp.shape[0],1))
    
    sen_pred = []
    
    for i in range(max_len):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output, bert=bert_enc)
        
        predictions, attention_weights = transformer(encoder_input, 
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)
        
        
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
        sen_pred.append(predictions)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        
        output = tf.concat([output, predicted_id], axis=-1)
        
    sen_pred = tf.concat(sen_pred[:-1], axis=1) # (batch_size, max_length_targ-1, vocab_size)
    
    loss = utils.loss_function(targ_real, sen_pred, loss_object)
    
    val_loss(loss)
    val_accuracy(targ_real, sen_pred)
    
    return output[:,1:], attention_weights

#######

def inference(data, start_tok, transformer, bert_enc=False):
    
    val_loss.reset_states()
    val_accuracy.reset_states()
    
    attention_weights = []
    pred = []
    for (batch,(inp,targ)) in enumerate(data):
        
        output, att_weights = inference_step(inp, targ, start_tok, transformer, bert_enc=False)
        
        attention_weights.append(att_weights)
        pred.append(output)
    
    return pred, attention_weights



###########



def train_model(path):

    df_train, df_test = utils.loadDataset(path, 2000, 69)

    lang, vocab_size, start_tok, end_tok, (inp_tensor_train, targ_tensor_train, inp_tensor_test, targ_tensor_test) = psl_tokenizer(
        TOKENIZER, df_train, df_test, input_language, targ_language, pad="post"
    )

    config.add_section('TEMP')
    config.set('TEMP','start_tok', str(start_tok))
    config.set('TEMP','end_tok', str(end_tok))

    with open(config_file,'w') as cfg:
        config.write(cfg)

    dev_size = utils.set_dev_size(val_size,inp_tensor_train.shape[0])

    X_train, X_val, y_train, y_val = train_test_split(
        inp_tensor_train,
        targ_tensor_train,
        test_size=dev_size,
        random_state=random_state
    )


    train_dataset, val_dataset, test_dataset = utils.create_tf_dataset(
        (X_train, y_train),(X_val, y_val),(inp_tensor_test, targ_tensor_test),
        BATCH_SIZE_TRAIN, BATCH_SIZE_VAL, BATCH_SIZE_TEST, verbose=True
    )

    misses = []

    if embedding is 'bert':
        embedding_dim = 768
        #embedding_matrix = loadBert()
    elif embedding is 'glove':
        embedding_matrix, misses, embedding_index = Embeddings.loadGlove( d_model, vocab_size, lang, tok_size)
    elif embedding is 'fasttext':
        embedding_matrix, misses = Embeddings.loadFasttext(200000, lang, vocab_size, subword=False)
    elif embedding is 'bpe':
        embedding_matrix = Embeddings.loadBpeEmb(lang)
        d_model = 100  
    elif embedding is 'uniform':
        embedding_matrix = 'uniform'



    @tf.function()
    def train_step(inp, tar, transformer, optimizer, bert=False):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp,bert=bert_enc)
        
        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp, 
                                        True, 
                                        enc_padding_mask, 
                                        combined_mask, 
                                        dec_padding_mask)
            loss = utils.loss_function(tar_real, predictions, loss_object)

        gradients = tape.gradient(loss, transformer.trainable_variables)    
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
        
        train_loss(loss)
        train_accuracy(tar_real, predictions)


    transformer = Transformer(num_layers, d_model, num_heads, dff,
                          vocab_size, vocab_size, 
                          pe_input=vocab_size, 
                          pe_target=vocab_size,
                          rate=dropout_rate,
                          emd_matrix=embedding_matrix,
                          train_emb = train_emb,
                          bert = bert_enc
                         )
    
    learning_rate = utils.CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

    checkpoint_path = "./checkpoints/train"

    ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

    history = {'train_loss':[],
           'val_loss':[],
           'train_acc':[],
           'val_acc':[]
          }


    for epoch in range(EPOCHS):
        start = time.time()
    
        train_loss.reset_states()
        train_accuracy.reset_states()
        

    
        # inp -> english, tar -> psl
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar, transformer, optimizer, bert=False)
            if batch % 100 == 0:
                print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))
        

        if (epoch + 1) % 1 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                 ckpt_save_path))


        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            train_loss.result(), 
                                                            train_accuracy.result()))
        
        inference(val_dataset, start_tok, transformer) # calculate validation loss
        print ('Epoch {} val Loss {:.4f} val Accuracy {:.4f}'.format(epoch + 1,
                                                                    val_loss.result(), 
                                                                    val_accuracy.result()))

        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
        
        history['train_loss'].append(train_loss.result())
        history['train_acc'].append(train_accuracy.result())
        history['val_loss'].append(val_loss.result())
        history['val_acc'].append(val_accuracy.result())

    return history



  