


@tf.function
def train_step(inp, targ, enc_hidden):
    # enc_hidden: is first hidden state of encoder, a vector of zeros
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
    
        # Encoder's final hidden state will be used as decoders first hidden state
        dec_hidden = enc_hidden
        
        dec_input = tf.expand_dims([start_tok] * BATCH_SIZE_TRAIN, 1)


        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)
            train_accuracy(targ[:, t], predictions)
            
            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    
    
    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


