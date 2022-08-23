from sklearn import linear_model
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier
from typing import Dict, Any, Union, List
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd
from pathlib import Path
import logging
import joblib

logger = logging.getLogger(__name__)

SklearnModel = Union[sklearn.base.RegressorMixin, sklearn.base.ClassifierMixin]


class FilterInputFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features: List[str]):
        self.features = features

    def fit(self, X: pd.DataFrame, y=None) -> 'FilterInputFeatures':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.features]



def parse_model_from_config(config: Dict[str, Any]) -> SklearnModel:
      """
      parse from model config file
      config_file: Dict[str, Any]
            {
            model: str (registered in DISPATCHER)
            params:
                  hyper parameters of the model
            }
      """

      model_name = config['model']
      params = config['parameters']
      assert model_name in DISPATCHER, '{} model is not in registered models'.format(model_name)
      model_instance = DISPATCHER[model_name]
      model: SklearnModel = model_instance(**params)
      return model


def build_pipeline_from_config(config: Dict[str, Any], features: List[str]) -> Pipeline:
      """
      returns a pipeline to record the input features
      """

      model = parse_model_from_config(config)
      input_feat_tmf = FilterInputFeatures(features)
      pipeline = Pipeline([
            ('input', input_feat_tmf),
            ('model', model)
      ])
      return pipeline


def save_multiple_models(models: List[Pipeline],
                         dir: str) -> None:
      """
      save model in pkl format
      """
      dir = Path(dir)
      dir.mkdir(parents=True, exist_ok=True)
      for i, model in enumerate(models):
            output_model_path = dir / f'model_{i}.pkl'
            logger.info(f'writing model {output_model_path}')
            joblib.dump(model, output_model_path)


DISPATCHER = {
    'logreg': linear_model.LogisticRegression,
    'lgbm': LGBMClassifier,
    'xgb': XGBClassifier,
    'xgb_rf': XGBRFClassifier
}
