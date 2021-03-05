
import tensorflow as tf

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