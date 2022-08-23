# def optimize_lgbm_parameters()
from typing import List, Dict, Any
import pandas as pd
from src import train, util, inference
import itertools
import numpy as np
import os
import cfg
from pathlib import Path

def get_basic_config(scale_pos_weight: float) -> Dict[str, Any]:
    base_config = {
        "model": "lgbm",
        "parameters": {
            "colsample_bytree": .3,
            "scale_pos_weight": scale_pos_weight,
            "n_estimators": 50
        }
    }

    return base_config


def optimize_lgbm_hp(data, features, target_name, cv_dir):
    _hp_scores = []
    scale_pos_weight = (data[target_name] == 0).sum() / data[target_name].sum()
    
    scale_pos_weight_scape = np.arange(1, scale_pos_weight, scale_pos_weight/10).tolist() + [scale_pos_weight]
    hyper_space = {
                'colsample_bytree': np.arange(0.1, 1., 0.25).tolist(),
                'scale_pos_weight': scale_pos_weight_scape
                }

    lgbm_params = [dict(zip(hyper_space.keys(), params)) for params in itertools.product(*hyper_space.values())]


    lgbm_configs = [{'model': 'lgbm', 'parameters': params} for params in lgbm_params]

    print(f'n of iterations: {len(lgbm_configs)}')
    for i, hp_config in enumerate(lgbm_configs):
        print(f'config: {i}/{len(lgbm_configs)}')
        print(hp_config)
        hp_models =  train.train_cv_from_config(data,
                                            hp_config,
                                            features=features,
                                            target=target_name,
                                            cv_dir=cv_dir,
                                            train_model_fn=train.train_lgbm,
                                            use_early_stopping=True)
        
        
        hp_scores = inference.cross_validation_inference(data, target_name, hp_models, cv_dir)
        
        parameters = pd.Series(hp_config['parameters'])
        parameters['loss'] = np.mean(hp_scores)
        _hp_scores.append(parameters)
        print('\n\n\n\n')

    hp_scores = pd.DataFrame(_hp_scores).sort_values(by='loss')

    return hp_scores


def run_optimization(data: pd.DataFrame,
         features: List[str],
         target_name: str,
         cv_dir: str):
    
    scale_pos_weight = (data[target_name] == 0).sum() / data[target_name].sum()
    base_config = get_basic_config(scale_pos_weight)

    util.pretty_print_config(base_config)
    base_models = train.train_cv_from_config(data,
                                         base_config,
                                         features=features,
                                         target=target_name,
                                         cv_dir=cv_dir)


    base_scores = inference.cross_validation_inference(data, target_name, base_models, cv_dir)

    base_scores_stats = base_scores.describe()

    print('base scores for target: {}:'.format(target_name))
    print(base_scores_stats)
    # selecting only important features

    permutation_fi = inference.compute_permutation_importance(
    data, target_name=target_name, models=base_models,
    features=features, cv_dir=cv_dir
)

    neg_permutation_fi = permutation_fi.copy()
    neg_permutation_fi['importance'] *= -1
    agg_permutation_fi = neg_permutation_fi.groupby('feature')['importance'].agg(['mean', 'std'])
    agg_permutation_fi['lower_bound']  = agg_permutation_fi['mean'] - 1.96 * agg_permutation_fi['std']
    agg_permutation_fi = agg_permutation_fi.sort_values(by='mean', ascending=False)


    relevant_features = agg_permutation_fi[agg_permutation_fi['mean'] > 0.001].index.to_list()
    print(f'# of relevant features: {len(relevant_features)}')
    print(relevant_features)
    fi_models = train.train_cv_from_config(data,
                                         base_config,
                                         features=relevant_features,
                                         target=target_name,
                                         cv_dir=cv_dir)

    fi_scores = inference.cross_validation_inference(data, target_name, fi_models, cv_dir)
    fi_scores_stats = fi_scores.describe()

    print('fi scores for target: {}:'.format(target_name))
    print(fi_scores_stats)


    hp_scores = optimize_lgbm_hp(data, relevant_features, target_name, cv_dir)


    print(hp_scores.head(10).to_markdown())


    best_config = {
    'model': 'lgbm',
    'parameters': hp_scores.drop('loss', axis=1).iloc[0].to_dict()
    }
    util.pretty_print_config(best_config)

    early_stop_models = train.train_cv_from_config(data,
                                         best_config,
                                         features=relevant_features,
                                         target=target_name,
                                         cv_dir=cv_dir,
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
                                         target=target_name,
                                         cv_dir=cv_dir,
                                       )
    opt_tree_scores = inference.cross_validation_inference(data, target_name, opt_tree_models, cv_dir)
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
    return output_config, scores_summary, hp_scores



def cli():
    # set paths
    current_dir = os.path.dirname(__file__)
    config_dir = os.path.join(current_dir, 'models')
    output_dir = os.path.join(current_dir, 'artifacts')
    os.makedirs(config_dir, exist_ok=True)

    
    data_dir = Path(cfg.DATA_DIR) 
    cv_dir = data_dir / 'cv_index'
    proccesed_dir = data_dir / 'proccessed'
    raw_dir = data_dir / 'raw'

    # reading data
    pd_features = pd.read_csv(proccesed_dir / 'max_norm_agg_ts_features.csv', index_col='sample_id')
    pd_train_target = pd.read_csv(raw_dir / 'train_labels.csv', index_col='sample_id')
    data = pd.concat([pd_train_target, pd_features], axis=1)

    features = pd_features.columns.to_list()
    # run for every target
    for target_name in cfg.TARGETS[2:]:
        output_config, scores_summary, hp_scores = (
            run_optimization(data, features, target_name, cv_dir)
        )


        config_output_path = os.path.join(config_dir, f'{target_name}.yml')
        target_output_dir = os.path.join(output_dir, target_name)
        os.makedirs(target_output_dir, exist_ok=True)


        util.write_yml(output_config, config_output_path)
        scores_summary.to_csv(os.path.join(target_output_dir, 'scores_summary.csv'))
        hp_scores.to_csv(os.path.join(target_output_dir, 'hp_scores.csv'), index=False)
    

if __name__ == '__main__':
    cli()

