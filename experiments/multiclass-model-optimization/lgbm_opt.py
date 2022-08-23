from typing import List, Dict, Any
import pandas as pd
from src import train, util, inference
import itertools
import numpy as np
import os
import cfg
from pathlib import Path
from src.data import get_cv_paths, join_dataframe_columns
from src.data import setup_directories

def predict_multiclass_fn(model, test_data):
    probs = model.predict_proba(test_data)
    return probs[:, 2:].sum(axis=1)


def build_multiclassifcation_target(data: pd.DataFrame, target_name: str) -> pd.Series:
    n_act_targets = data[cfg.TARGETS].sum(axis=1)
    only_target = (n_act_targets == 1) & (data[target_name] == 1)
    multiple_targets = (n_act_targets > 1) & (data[target_name] == 1)
    other_targets = (n_act_targets > 0) & (data[target_name] == 0)
    none_targets = (n_act_targets  == 0) & (data[target_name] == 0)
    
    assert (only_target & multiple_targets).sum() == 0
    assert (only_target & other_targets).sum() == 0
    assert (only_target & none_targets).sum() == 0
    
    multiclass_target = np.full(len(data), np.nan)
    multiclass_target[none_targets] = 0
    multiclass_target[other_targets] = 1
    multiclass_target[only_target] = 2
    multiclass_target[multiple_targets] = 3
    
    assert np.isnan(multiclass_target).sum() == 0
    
    return multiclass_target.astype(np.int64)


def run_optimization(data: pd.DataFrame,
         features: List[str],
         target_name: str,
         cv_dir: str):

    cv_paths = get_cv_paths(cv_dir, target_name)
    
    base_config = {
        'model': 'lgbm',
        'parameters':
            {
                'class_weight': 'balanced',
                'n_estimators': 50,
                'colsample_bytree': 0.3
            }
        }


    multiclass_target = f'{target_name}_multiclass'

    util.pretty_print_config(base_config)
    base_models = train.train_cv_from_config(
        data,
        base_config,
        features=features,
        target=multiclass_target,
        cv_paths=cv_paths)


    base_scores = inference.cross_validation_inference(
        data, target_name=target_name,
        models=base_models,
        cv_paths=cv_paths,
        predict_fn=predict_multiclass_fn)

    base_scores_stats = base_scores.describe()

    print('base scores for target: {}:'.format(target_name))
    print(base_scores_stats)
    # selecting only important features

    permutation_fi = inference.compute_permutation_importance(
    data, target_name=target_name,
    models=base_models,
    features=features,
    cv_paths=cv_paths,
    predict_fn=predict_multiclass_fn
)

    neg_permutation_fi = permutation_fi.copy()
    neg_permutation_fi['importance'] *= -1
    agg_permutation_fi = neg_permutation_fi.groupby('feature')['importance'].agg(['mean', 'std'])
    agg_permutation_fi['lower_bound']  = agg_permutation_fi['mean'] - 1.96 * agg_permutation_fi['std']
    agg_permutation_fi = agg_permutation_fi.sort_values(by='mean', ascending=False)


    relevant_features = agg_permutation_fi[agg_permutation_fi['mean'] > 0.].index.to_list()
    print(f'# of relevant features: {len(relevant_features)}')
    print(relevant_features)
    fi_models = train.train_cv_from_config(
        data,
        base_config,
        features=relevant_features,
        target=multiclass_target,
        cv_paths=cv_paths)


    fi_scores = inference.cross_validation_inference( 
        data, target_name=target_name,
        models=fi_models,
        cv_paths=cv_paths,
        predict_fn=predict_multiclass_fn)

    fi_scores_stats = fi_scores.describe()

    print('fi scores for target: {}:'.format(target_name))
    print(fi_scores_stats)


    # no hyper opt optimization, only n_estimators
    best_config = {
        'model': 'lgbm',
        'parameters':
            {
                'class_weight': 'balanced',
                'n_estimators': 500,
                'colsample_bytree': 0.3
            }
        }
    util.pretty_print_config(best_config)
    early_stop_models = train.train_cv_from_config(data,
                                         best_config,
                                         features=relevant_features,
                                         target=multiclass_target,
                                         cv_paths=cv_paths,
                                         train_model_fn=train.train_lgbm,
                                         use_early_stopping=True)
    
    best_n_trees = int(np.mean([pipeline['model'].best_iteration_
                                for pipeline in early_stop_models]))


    best_opt_tree_config = best_config.copy()
    best_opt_tree_config['parameters']['n_estimators'] = best_n_trees
    util.pretty_print_config(best_opt_tree_config)

    opt_tree_models = train.train_cv_from_config(data,
                                         best_opt_tree_config,
                                         features=relevant_features,
                                         target=multiclass_target,
                                         cv_paths=cv_paths,
                                       )
    opt_tree_scores = inference.cross_validation_inference(
        data, target_name=target_name,
        models=opt_tree_models,
        cv_paths=cv_paths,
        predict_fn=predict_multiclass_fn)
        
    opt_tree_scores_stats = opt_tree_scores.describe()


    print('final optimizal scores for target: {}:'.format(target_name))
    print(opt_tree_scores_stats)


    _scores_summary = [base_scores_stats, fi_scores_stats, opt_tree_scores_stats]
    scores_summary = pd.DataFrame(
        _scores_summary,
        index=['base', 'after_feature_opt', 'after_hp_opt']
        
    )

    print(scores_summary.to_markdown)


    output_config = {'model_config': best_opt_tree_config,
                     'features': relevant_features} 
    
    util.pretty_print_config(output_config)
    return output_config, scores_summary#, agg_permutation_fi



def cli():
    # set paths
    current_dir = os.path.dirname(__file__)
    config_dir = os.path.join(current_dir, 'models', 'lgbm')
    output_dir = os.path.join(current_dir, 'artifacts', 'lgbm')
    os.makedirs(config_dir, exist_ok=True)
    dirs = setup_directories(cfg.DATA_DIR, create_dirs=True)
    raw_dir = Path(dirs['raw'])
    train_dir = Path(dirs['train'])
    cv_dir = Path(dirs['cv']['final-validation'])
    # read train labels
    pd_train_target = pd.read_csv(raw_dir / 'train_labels.csv', index_col='sample_id')
    # read train labels
    pd_multclass_target = pd.read_csv(train_dir / 'multiclass.csv', index_col='sample_id')
    # reading data
    pd_agg_features = pd.read_csv(train_dir / 'mz_agg_features_drop_correlated.csv', index_col='sample_id')

    pd_agg_features.head()

    pd_cluster_features = pd.read_csv(train_dir / 'ae_clusters.csv', index_col='sample_id')

    pd_cluster_features.head()

    pd_sample_features = pd.read_csv(train_dir / 'sample_features.csv', index_col='sample_id')

    pd_features = pd.concat((pd_sample_features, pd_agg_features, pd_cluster_features), axis=1)

    feature_names = pd_features.columns.to_list()


    data = pd.concat((pd_train_target, pd_multclass_target, pd_features), axis=1)

    # run for every target
    for target_name in cfg.TARGETS:
        output_config, scores_summary = (
            run_optimization(data, feature_names, target_name, cv_dir)
        )


        config_output_path = os.path.join(config_dir, f'{target_name}.yml')
        target_output_dir = os.path.join(output_dir, target_name)
        os.makedirs(target_output_dir, exist_ok=True)


        util.write_yml(output_config, config_output_path)
        scores_summary.to_csv(os.path.join(target_output_dir, 'scores_summary.csv'))
        # fi.to_csv(os.path.join(target_output_dir, 'fi.csv'), index=False)
    

if __name__ == '__main__':
    cli()

