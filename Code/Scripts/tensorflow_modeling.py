import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.metrics import auc, f1_score, roc_auc_score, recall_score, precision_score, brier_score_loss
import functools
import os

vocab_dir = '../../data/diabetes_vocab/'

'''
From Tensorflow Probability Regression tutorial  https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_Layers_Regression.ipynb    
'''

def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2*n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
            tfp.distributions.Normal(loc=t[..., :n],
                                     scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])


def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
            tfp.distributions.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1)),
    ])

def demo(feature_column, example_batch):
    feature_layer = tf.keras.layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch))
    return feature_layer(example_batch)

def calculate_stats_from_train_data(df, col):
    mean = df[col].describe()['mean']
    std = df[col].describe()['std']
    return mean, std

def create_tf_numerical_feature_cols(numerical_col_list, train_df):
    tf_numeric_col_list = []
    for c in numerical_col_list:
        mean, std = calculate_stats_from_train_data(train_df, c)
        tf_numeric_feature = create_tf_numeric_feature(c, mean, std)
        tf_numeric_col_list.append(tf_numeric_feature)
    return tf_numeric_col_list


#adapted from https://www.tensorflow.org/tutorials/structured_data/feature_columns
def df_to_dataset(df, predictor,  batch_size=32):
    df = df.copy()
    labels = df.pop(predictor)
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds


def write_vocabulary_file(vocab_list, field_name, default_value, vocab_dir=vocab_dir):
    output_file_path = os.path.join(vocab_dir, str(field_name) + "_vocab.txt")
    vocab_list = np.insert(vocab_list, 0, default_value, axis=0) 
    df = pd.DataFrame(vocab_list).to_csv(output_file_path, index=None, header=None)
    return output_file_path


def show_group_stats_viz(df, group):
    print(df.groupby(group).size().plot(kind='barh'))


def build_vocab_files(df, categorical_column_list, default_value='00'):
    vocab_files_list = []
    for c in categorical_column_list:
        v_file = write_vocabulary_file(df[c].unique(), c, default_value)
        vocab_files_list.append(v_file)
    return vocab_files_list
          
      
def create_tf_categorical_feature_cols(high_kardinality, low_kardinality, vocab_dir=vocab_dir):
    dims = 30
    output_tf_list = []
    for c in high_kardinality:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")        
        cat_vocab = tf.feature_column.categorical_column_with_vocabulary_file(
            key=c, vocabulary_file = vocab_file_path, num_oov_buckets=1)
        
        tf_categorical_feature_column = tf.feature_column.embedding_column(cat_vocab, dimension=dims)
        
        output_tf_list.append(tf_categorical_feature_column)
        
    for col in low_kardinality:
        vocab_file_path = os.path.join(vocab_dir,  col + "_vocab.txt")        
        cat_vocab = tf.feature_column.categorical_column_with_vocabulary_file(
            key=col, vocabulary_file = vocab_file_path, num_oov_buckets=1)
        tf_categorical_feature_column = tf.feature_column.indicator_column(cat_vocab)
        output_tf_list.append(tf_categorical_feature_column)        
    return output_tf_list
 

def normalize_numeric_with_zscore(col, mean, std):
    return (col - mean)/std


def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    tf_numeric_feature = tf.feature_column.numeric_column(key=col, default_value=default_value, 
                                                          normalizer_fn=normalizer, dtype=tf.float64)
    return tf_numeric_feature


def get_mean_std_from_preds(diabetes_yhat):
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s


def get_binary_prediction(df, col, days):    
    binary_prediction = df[col].apply(lambda x: 1 if x >= days else 0).values
    return binary_prediction


def add_pred_to_test(test_df, pred_np, demo_col_list, days):
    test_df = test_df.copy()
    for c in demo_col_list:
        test_df[c] = test_df[c].astype(str)
    test_df['score'] = pred_np
    test_df['label_value'] = test_df['time_in_hospital'].apply(lambda x: 1 if x >=days else 0)
    return test_df

# AUC, F1, precision and recall
def print_metric_scores(pred_test_df, print_results=True):
    bs_loss = brier_score_loss(pred_test_df['score'], pred_test_df['label_value'])
    
    try:
        auc = roc_auc_score(pred_test_df['label_value'], pred_test_df['score'])
    except:
        auc = 0
        
    f1 = f1_score(pred_test_df['label_value'], pred_test_df['score'], average='weighted')
    precision = precision_score(pred_test_df['label_value'], pred_test_df['score'], average='micro')
    recall = recall_score(pred_test_df['label_value'], pred_test_df['score'], average='micro')
    
    if print_results:
        print("Brier score :", bs_loss)
        print("AUC score : ",auc)
        print("F1 score : ", f1)
        print("Precision score: ", precision)
        print("Recall score : ", recall)
     
    else:
        pass
    
    return bs_loss, auc, f1, precision, recall