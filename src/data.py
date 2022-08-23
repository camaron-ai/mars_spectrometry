from typing import Dict, List
import pandas as pd
import os
from collections import OrderedDict
import re
import numpy as np

def read_multiple_sample_features(sample_id_files: Dict[str, str]) -> str:
    return pd.concat([
        pd.read_csv(path).assign(sample_id=sample_id)
        for sample_id, path in sample_id_files.items()
    ])


def get_features_path_from_metadata(pd_metadata: pd.DataFrame, raw_dir: str = None) -> Dict[str, str]:
    sample_id_relpaths = pd_metadata['features_path']
    if raw_dir:
        sample_id_relpaths = sample_id_relpaths.apply(lambda p: os.path.join(raw_dir, p))
    
    return OrderedDict(sample_id_relpaths.to_dict())


def join_dataframe_columns(df: pd.DataFrame) -> None:
    """join dataframe columns inplace"""
    columns_names = [name if name else '' for name in df.columns.names]
    joined_columns = ['_'.join(map(lambda vs: '-'.join(map(str, vs)), zip(columns_names, col)))
                      for col in df.columns]
    joined_columns = [re.sub('[^A-Za-z0-9_]+', '', x) for x in joined_columns]
    df.columns = joined_columns


def get_cv_paths(cv_dir: str, target_name: str) -> List[str]:
    target_cv_dir = os.path.join(cv_dir, target_name) 
    cv_paths = sorted([os.path.join(target_cv_dir, filename)
                         for filename in os.listdir(target_cv_dir)])

    return cv_paths



def setup_directories(data_dir: str, create_dirs: bool = True) -> Dict[str, str]:
    raw_dir = os.path.join(data_dir, 'raw')
    processed = os.path.join(data_dir, 'processed')
    cv_dir = os.path.join(data_dir, 'cv_index')

    train_dir = os.path.join(processed, 'train')
    valid_dir = os.path.join(processed, 'valid')
    test_dir = os.path.join(processed, 'test')
    submission_dir = 'submission'

    if create_dirs:
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(valid_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(submission_dir, exist_ok=True)
    
    return {'raw': raw_dir, 
            'train': train_dir,
            'valid': valid_dir,
            'test': test_dir,
            'cv': 
                {
                    'validation': os.path.join(cv_dir, 'validation'),
                    'final-validation': os.path.join(cv_dir, 'cv-model'),
                    'test': os.path.join(cv_dir, 'cv-model-test')

                },
            'submission': submission_dir,
            }


def build_dataframe_as(array: np.ndarray, df: pd.DataFrame):
    assert array.shape == df.shape
    return pd.DataFrame(array, index=df.index, columns=df.columns)