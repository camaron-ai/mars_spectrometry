import mlflow
from typing import Dict, Any
from datetime import datetime as dt
from src.util import allign_dictionary_subclasses


DATE_FORMAT: str = '%Y-%m-%d-%H-%M-%S'

def str_now():
    return dt.now().strftime(DATE_FORMAT)


def track_experiment(run_name: str, 
                     mlflow_experiment: str,
                     config: Dict[str, Any],
                     scores: Dict[str, float],
                     artifacts_dir: str = None,
                     ):
    mlflow.set_experiment(mlflow_experiment)
    parameters = allign_dictionary_subclasses(config)
    with mlflow.start_run(run_name=run_name):
        # log parameters
        mlflow.log_params(parameters)
        # log metrics
        mlflow.log_metrics(scores)
        # log all artifacts
        if artifacts_dir:
            mlflow.log_artifacts(artifacts_dir)