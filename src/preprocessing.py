import numpy as np
from typing import Dict, Callable
import pandas as pd
from tqdm.auto import tqdm
from collections import Counter



def drop_mz_values(df: pd.DataFrame,
                   drop_fractional: bool = True,
                   remove_above_100: bool = True) -> pd.DataFrame:
    if drop_fractional:
        df = df[df["m/z"].transform(round) == df["m/z"]].copy()

    df['m/z'] = np.round(df['m/z']).astype(np.int64)

    # drop carrier gas
    df = df[df["m/z"] != 4]

    if remove_above_100:
        df = df[df["m/z"] < 100]
        assert df['m/z'].max() < 100

    # drop mz values with not enough sampling
    mz_sampling = df['m/z'].value_counts()
    mz_sampling = mz_sampling / mz_sampling.max()
    # drop mz with less 0.05% observations
    mz_to_keep = mz_sampling.index[mz_sampling >= 0.05]
    df = df[df['m/z'].isin(mz_to_keep)]
    return df
        
    

def join_sambtest_sampling(data: pd.DataFrame) -> pd.DataFrame:
    mz_counts = Counter()
    discrete_time = []
    count = 0
    data = data.sort_values(by=['time'])
    for mz_value in data['m/z']:
        if mz_counts[mz_value] == count:
            count += 1

        mz_counts[mz_value] = count
        discrete_time.append(count)
    
    
    data['discrete_time'] = discrete_time

    time_temp_mapper = data.groupby('discrete_time')[['temp', 'time']].max()
    data.drop(['temp', 'time'], axis=1, inplace=True)
    data = data.merge(time_temp_mapper, on=['discrete_time'], how='left')
    data.drop(['discrete_time'], axis=1, inplace=True)
    return data


def preprocess_sample(
    sample_data: pd.DataFrame,
    drop_fractional: bool = True,
    remove_above_100: bool = True,
    is_sam_testbed: bool = False) -> pd.DataFrame:
    sample_data = drop_mz_values(
        sample_data,
        remove_above_100=remove_above_100,
        drop_fractional=drop_fractional)

    if is_sam_testbed:
        sample_data = join_sambtest_sampling(sample_data)

    return sample_data



def apply_preprocessing_fn(
    files: Dict[str, str],
    pd_metadata: pd.DataFrame,
    processing_fn: Callable,
    remove_above_100: bool = True,
    drop_fractional: bool = True,
    **kwargs) -> Dict[str, pd.DataFrame]:
    output_features = {}


    for sample_id, path in tqdm(files.items()):
        sample_data = pd.read_csv(path)
        sample_data = preprocess_sample(
            sample_data, drop_fractional=drop_fractional,
            remove_above_100=remove_above_100,
            is_sam_testbed=pd_metadata.loc[sample_id, 'instrument_type'] == 'sam_testbed'
        )
        features = processing_fn(sample_data, **kwargs)
        output_features[sample_id] = features
    
    return output_features
    



def post_processing_prediction(data: pd.DataFrame, target_name, yhat):
    if target_name not in ['oxychlorine', 'chloride']:
        return yhat
    
    assert 'sample_mol_ion_less99' in data.columns
    yhat[data['sample_mol_ion_less99'].astype(np.bool_)] = 0.
    return yhat


