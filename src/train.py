"""

base train with sklearn api
nothing fancy implemeted here

"""
from typing import Dict, Any, List, Callable, Optional
import pandas as pd
import time
import logging
from src.data import get_cv_paths
from src.models import build_pipeline_from_config
from sklearn.pipeline import Pipeline
from src.model_selection import get_train_test_tuple_from_split

logger = logging.getLogger(__name__)


def train_sklearn_api_model(model: Pipeline,
      train_data: pd.DataFrame,
      features: List[str],
      target: str,
      valid_data: Optional[pd.DataFrame] = None) -> Pipeline:
      """base train function for sklearn api models"""
      start_time = time.time()
      model.fit(train_data.loc[:, features], train_data.loc[:, target])
      elapsed_time = (time.time() - start_time) / 60
      logger.info(f'elapsed training time: {elapsed_time:.3f} min')
      return model



def train_model_from_config(
      train_data: pd.DataFrame,
      model_config: Dict[str, Any],
      features: List[str],
      target: str,
      train_model_fn: Optional[Callable] = train_sklearn_api_model,
      valid_data: Optional[pd.DataFrame] = None,
      **train_params) -> Pipeline:

    model = build_pipeline_from_config(model_config, features=features)

    model = train_model_fn(model, train_data, features, target,
                           valid_data=valid_data, **train_params)

    return model



def train_cv_from_config(
      data: pd.DataFrame,
      model_config: Dict[str, Any],
      features: List[str],
      target: str,
      cv_paths: str,
      train_model_fn: Optional[Callable] = train_sklearn_api_model,
      **train_params) -> List[Pipeline]:

      models = []
      for fold, cv_path in enumerate(cv_paths, start=1):
            logger.info(f'fold={fold}/{len(cv_paths)}')
            logger.info(f'reading cv index from {cv_path}')
            train_data, test_data = get_train_test_tuple_from_split(data, cv_path)
            model = train_model_from_config(train_data, model_config,
                                          features=features, target=target,
                                          train_model_fn=train_model_fn,
                                          valid_data=test_data,
                                          **train_params)
            models.append(model)

      return models


def train_one_vs_rest_model(
      data: pd.DataFrame,
      model_config: Dict[str, Any],
      features: List[str],
      targets: List[str],
      cv_dir: str,
      train_model_fn: Optional[Callable] = train_sklearn_api_model,
      **train_params) -> Dict[str, List[Pipeline]]:

      models = {}
      for target in targets:
            logger.info(f'target={target}')
            target_models = (
                  train_cv_from_config(data,
                  model_config,
                  features,
                  target,
                  cv_dir,
                  train_model_fn,
                  **train_params))
            
            models[target] = target_models
      
      return models




def train_lgbm(
      model: Pipeline,
      train_data: pd.DataFrame,
      features: List[str],
      target: str,
      valid_data: Optional[pd.DataFrame] = None,
      use_early_stopping: int = False) -> Pipeline:
      """
      helper function to train lgbm model
      """

      eval_sets = [(train_data[features], train_data[target])]
      eval_names = ['train']

      if valid_data is not None:
            eval_sets.append((valid_data[features], valid_data[target]))
            eval_names.append('valid')

      
      early_stopping_rounds: int = (model['model'].n_estimators // 10
                                    if use_early_stopping else None)
      start_time = time.time()
      model.fit(train_data.loc[:, features],
                train_data.loc[:, target],
                model__early_stopping_rounds=early_stopping_rounds,
                model__eval_set=eval_sets,
                model__eval_names=eval_names,
                model__verbose=early_stopping_rounds,
            )
      elapsed_time = (time.time() - start_time) / 60
      logger.info(f'elapsed training time: {elapsed_time:.3f} min')
      return model