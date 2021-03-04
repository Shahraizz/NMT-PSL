

optimizer = Adam()

loss_object = SparseCategoricalCrossentropy(
    from_logits=True, reduction='none'
)

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

def trainable_variable_len(e):
    count = 0
    for x in e.trainable_variables:
        count += x.numpy().size
        
    return count
