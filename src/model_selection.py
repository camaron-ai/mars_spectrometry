from typing import Tuple
import pandas as pd


def get_train_test_tuple_from_split(
    data: pd.DataFrame,
    split_csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

    split_csv = pd.read_csv(split_csv_path, index_col='sample_id')
    assert len(split_csv) == len(data)
    assert (data.index == split_csv.index).all(), 'index mismatch'

    data['_is_test'] = split_csv['is_test'].to_numpy()
    train_data = data.query('_is_test == False').copy()
    test_data = data.query('_is_test == True').copy()

    data.drop('_is_test', axis=1, inplace=True)
    train_data.drop('_is_test', axis=1, inplace=True)
    test_data.drop('_is_test', axis=1, inplace=True)

    return train_data, test_data