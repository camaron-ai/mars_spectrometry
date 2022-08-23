import pandas as pd
from sklearn.metrics import log_loss
import numpy as np
from typing import Dict, List
from src.model_selection import get_train_test_tuple_from_split
from sklearn.pipeline import Pipeline
import cfg
from sklearn.inspection import permutation_importance
from tqdm.auto import tqdm


def predict_proba_fn(model: Pipeline, test_data):
    return  model.predict_proba(test_data)[:, 1]


def cross_validation_inference(
    data: pd.DataFrame,
    target_name: str,
    models: List[Pipeline],
    cv_paths: List[str],
    predict_fn = predict_proba_fn) -> pd.Series:

    assert len(cv_paths) == len(models)
    scores = []
    for fold, cv_path in enumerate(cv_paths):
        _, test_data = get_train_test_tuple_from_split(data, cv_path)
        model = models[fold]
        prediction = predict_fn(model, test_data)
        scores.append(log_loss(test_data[target_name], prediction))
    
    return pd.Series(scores)


def compute_permutation_importance(
    data: pd.DataFrame,
    target_name: str,
    models: List[Pipeline],
    features: List[str],
    cv_paths: str,
    predict_fn = predict_proba_fn) -> pd.DataFrame:

    def clf_score_fn(estimator: Pipeline, X: pd.DataFrame, y: np.ndarray) -> float:
        yhat = predict_fn(estimator, X)
        return log_loss(y, yhat)

    _permutation_fi = []


    assert len(cv_paths) == len(models)
    for fold, cv_path in enumerate(tqdm(cv_paths)):
        _, test_data = get_train_test_tuple_from_split(data, cv_path)
        model = models[fold]


        per_imp = permutation_importance(model,
                                 test_data[features],
                                 test_data[target_name],
                                 scoring=clf_score_fn,
                                 n_jobs=-1,
                                 random_state=123,
        
        )
        fi = pd.DataFrame({'feature': features, 'importance': per_imp.importances_mean})
        _permutation_fi.append(fi)    

    return pd.concat(_permutation_fi)




def compute_performance_from_models(
    data: pd.DataFrame,
    models: Dict[str, List[Pipeline]],
    cv_paths: str,
    targets: List[str] = cfg.TARGETS) -> pd.DataFrame:
    _scores = {
            target_name: cross_validation_inference(data, target_name, models[target_name], cv_paths)
            for target_name in targets

    }

    pd_scores = pd.DataFrame(_scores)
    pd_scores.insert(0, 'avg_loss', pd_scores.mean(axis=1))
    return pd_scores


def compute_avg_prediction(
    data: pd.DataFrame,
    models: List[Pipeline],
    cv_paths: List[str],
    predict_fn = predict_proba_fn) -> pd.Series:

    assert len(cv_paths) == len(models)
    yhat = []
    for fold, cv_path in enumerate(cv_paths):
        _, test_data = get_train_test_tuple_from_split(data, cv_path)
        model = models[fold]
        fold_yhat = predict_fn(model, test_data)
        pd_fold_yhat = pd.Series(fold_yhat, index=test_data.index)
        yhat.append(pd_fold_yhat)
    
    return pd.concat(yhat).sort_index()

        
