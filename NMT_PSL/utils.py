
import tensorflow as tf
from sklearn.utils import shuffle
import pandas as pd

def create_tf_dataset(
    train_pair,dev_pair,test_pair,batch_size_train,
    batch_size_dev,batch_size_test, verbose=False):

    ## Dataset for train
    train_dataset = tf.data.Dataset.from_tensor_slices(train_pair).shuffle(len(train_pair[0]))
    train_dataset = train_dataset.batch(batch_size_train, drop_remainder=True)

    ## Dataset for validation
    val_dataset = tf.data.Dataset.from_tensor_slices(dev_pair)
    val_dataset = val_dataset.batch(batch_size_dev, drop_remainder=True)

    ## Dataset for test 
    test_dataset = tf.data.Dataset.from_tensor_slices(test_pair)
    test_dataset = test_dataset.batch(batch_size_test, drop_remainder=True)
    
    if verbose:
        print('Number of train Batches: {}'.format(
            len(list(train_dataset.as_numpy_iterator()))
        ))
        print('Number of val Batches: {}'.format(
            len(list(val_dataset.as_numpy_iterator()))
        ))
        print('Number of test Batches: {}'.format(
            len(list(test_dataset.as_numpy_iterator()))
        ))
    
    return train_dataset, val_dataset, test_dataset


def set_dev_size(around, total_size):
    for i in range(around,around+64+1):
        if (total_size- i) % 64 == 0:
            return i


def loadDataset(path, test_size, random_state):
    
    df = pd.read_excel(path, usecols=['English', 'PSL']).dropna()
    
    df = shuffle(df,random_state=random_state).reset_index(drop=True)

    train_data = df.iloc[:-test_size].reset_index(drop=True)

    test_data = df.iloc[-test_size:].reset_index(drop=True)
    
    return train_data, test_data



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)






def loss_function(real, pred, loss_object):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)