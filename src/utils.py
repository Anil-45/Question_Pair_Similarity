"""Utils"""

import os
import pandas as pd
import numpy as np
import pickle

def reduce_mem_usage(df):
    """Reduce memory usage of a dataframe

    Args:
        df : pd.DataFrame

    Returns:
        pd.DataFrame : Dataframe with reduced size
    """
    initial_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(initial_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    final_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(final_mem))
    print('Decreased by {:.1f}%'.format(100 * (initial_mem - final_mem) / initial_mem))
    
    return df

def save_dict(dictionary, path):
    """Save dictionary to a file.

    Args:
        dictionary (dict): dictionary
        path (String): path
    """
    with open(path, 'wb') as fp:
        pickle.dump(dictionary, fp)
        
def load_dict(path):
    """Load dictionary from a path.

    Args:
        path (String): path
    
    Returns:
        dict: Loaded dictionary
    """
    assert os.path.exists(path) == True, "File does not exist"
    with open(path, 'rb') as fp:
        dictionary = pickle.load(fp)
    return dictionary
    