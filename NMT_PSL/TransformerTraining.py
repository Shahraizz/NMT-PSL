from .TransformerModel import Transformer
from .TransformerModel import create_padding_mask
from .TransformerModel import create_look_ahead_mask
from .TransformerModel import create_masks


import numpy as np



def inference_step(inp, targ):
    
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
    
    loss = loss_function(targ_real, sen_pred)
    
    val_loss(loss)
    val_accuracy(targ_real, sen_pred)
    
    return output[:,1:], attention_weights
    #return tf.math.argmax(sen_pred, -1), attention_weights

#######

def inference(data):
    
    val_loss.reset_states()
    val_accuracy.reset_states()
    
    attention_weights = []
    pred = []
    for (batch,(inp,targ)) in enumerate(data):
        
        output, att_weights = inference_step(inp, targ)
        
        attention_weights.append(att_weights)
        pred.append(output)
    
    return pred, attention_weights



###########



@tf.function()
def train_step(inp, tar):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]
  
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp,bert=bert_enc)
  
  with tf.GradientTape() as tape:
    predictions, _ = transformer(inp, tar_inp, 
                                 True, 
                                 enc_padding_mask, 
                                 combined_mask, 
                                 dec_padding_mask)
    loss = loss_function(tar_real, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  
  train_loss(loss)
  train_accuracy(tar_real, predictions)

