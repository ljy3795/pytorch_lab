#-*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os, time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import namedtuple

def prepare_dataset(df, wide_cols, crossed_cols, embeddings_cols, continuous_cols, target,
    scale=False, seed=1):
    
    """
        This is the function for preparation of "Wide and Deep learning Recommendation" dataset
        
        Input args. :
            - df : raw dataset [Pandas DataFrame]
            - wide_cols : column list for the Wide part
            - crossed_cols : column list for the Wide part (value type : list(pair))
                                ex) [[col_a, col_b], [col_b, col_c]]
            - embeddings_cols : column list for the Wide part (value type : tuple)
                                ex) [(col_a, 5), (col_b, 10)]
            - continuous_cols : column list for the Wide part
            - target : target variable
            - scale : boolean for scaling of numerical variables
            - seed : random number seed for train/test split
            
        Return value : 
            - 
    """
    
    
    ############################################
    # raw dataset
    raw_df = df
    
    
    ############################################
    # columns for wide and deep parts
    ### For Wide and Crossed network (for memorization)
    wide_cols = wide_cols
    crossed_cols = crossed_cols
    
    ### For Deep network (embedding + continuous) (for generalization)
    embeddings_cols = embeddings_cols
    continuous_cols = continuous_cols
    
    embdding_dim = dict(embeddings_cols)
    embeddings_cols = list(embdding_dim.keys())

    deep_cols = embeddings_cols + continuous_cols
    
    
    ############################################
    # target variable
    target_Y = np.array(raw_df[target])

    
    ############################################
    # feature handling for wide and deep columns
    tmp_df = raw_df.copy()[list(set(wide_cols + deep_cols))]

    # Make crossed cols
    crossed_columns = []
    for cols in crossed_cols:
        tmp_col_nm = '-'.join(cols)
        tmp_df[tmp_col_nm] = tmp_df[cols].apply(lambda x: '-'.join(x), axis = 1)
        crossed_columns.append(tmp_col_nm)

        
    ############################################
    # encoding for categorical columns
    ## find categorical variables
    categorical_cols = list(tmp_df.select_dtypes(include = 'object').columns)
    
    # step 1:
    unique_values = dict()
    for col in categorical_cols:
        unique_values[col] = list(tmp_df[col].unique())

    # step 2:
    val_2_inx = dict()
    for k, v in unique_values.items():
        val_2_inx[k] = {v2: i for i, v2 in enumerate(unique_values[k])}

    # step 3:
    for k, v in val_2_inx.items():
        tmp_df[k] = tmp_df[k].apply(lambda x: val_2_inx[k][x])


    # only for deep cols, make embedding cols info
    encoding_dict = {k: v for k, v in val_2_inx.items() if k in deep_cols}
    embeddings_input = []
    for k, v in encoding_dict.items():
        embeddings_input.append([k, len(v), embdding_dim[k]]) # column name, number of unique items, embedding dims

        
    
    ############################################
    # normal scaling for numerical(continuous) variables
    if scale:
        scaler = StandardScaler()
        for col in continuous_cols:
            tmp_df[col] = scaler.fit_transform(tmp_df[col].values.reshape(-1,1))
            
    
    
    ############################################
    # split dataset to wide and deep part
    df_deep = tmp_df[deep_cols]
    deep_column_idx = {k:v for v,k in enumerate(df_deep.columns)}

    # for categorical variables in the Wide part, we are not going to use encoing variable instead one-hot encoding variables
    df_wide = tmp_df[wide_cols + crossed_columns]
    one_hot_cols = [c for c in wide_cols+crossed_columns if c in categorical_cols]
    df_wide = pd.get_dummies(df_wide, columns=one_hot_cols)# by converting categorical variables to one-hot dummys, columns length increased from 10 to 798
    
    
    
    
    ############################################
    # split dataset to train and test set
    x_train_deep, x_test_deep = train_test_split(df_deep.values, test_size = 0.3, random_state = seed)
    x_train_wide, x_test_wide = train_test_split(df_wide.values, test_size = 0.3, random_state = seed)
    y_train, y_test = train_test_split(target_Y, test_size = 0.3, random_state = seed)
    
    
    
    
    ############################################
    # split dataset to train and test set
    out_dataset = dict()
    train_set = namedtuple('train_set', 'wide, deep, labels')
    test_set = namedtuple('test_set', 'wide, deep, labels')
    out_dataset['train_set'] = train_set(x_train_wide, x_train_deep, y_train)
    out_dataset['test_set'] = test_set(x_test_wide, x_test_deep, y_test)
    out_dataset['embeddings_input'] = embeddings_input
    out_dataset['deep_column_idx'] = deep_column_idx
    out_dataset['encoding_dict'] = encoding_dict
    
    
    print("return key values {}".format(out_dataset.keys()))
    
    
    return out_dataset
