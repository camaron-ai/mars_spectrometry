import pandas as pd
from sklearn.metrics import log_loss
import numpy as np
from typing import Dict


def compute_scores(targets: pd.DataFrame, prediction: pd.DataFrame) -> pd.Series:
    assert np.isin(prediction.columns, targets.columns).all(), 'missing target columns in prediction'

    scores = {
        target_name: log_loss(targets[target_name].to_numpy(), prediction[target_name].to_numpy())
        for target_name in targets.columns
            }

    scores['avg_loss'] = np.mean(list(scores.values()))
    scores = pd.Series(scores)
    scores = scores.reindex(index=['avg_loss'] + sorted(targets.columns))
    return scores




def compute_scores_from_iterations(targets: pd.DataFrame, oof_prediction: Dict[str, np.ndarray]):
    scores = {
        target_name: [log_loss(targets[target_name].to_numpy(), yhat) for yhat in oof_prediction[target_name].T]
        for target_name in targets.columns
    }

    pd_scores = pd.DataFrame(scores)

    pd_scores.insert(0, 'avg_loss', pd_scores.mean(axis=1))
    return pd_scores