from pathlib import Path
import pandas as pd
import numpy as np
import click
import cfg
import os
import logging
from src.util import setup_logging
import shutil
from iterstrat.ml_stratifiers import RepeatedMultilabelStratifiedKFold
logger = logging.getLogger(__name__)



def write_cv_index(raw_dir: str,
                   cv_dir: str,
                   n_splits: int,
                   test_size: int,
                   add_validation: bool = False):
    logger.info('creating stratified cross validation index')
    # create cv output dir
    cv_dir = Path(cv_dir)
    raw_dir = Path(raw_dir)
    if os.path.exists(cv_dir):
        shutil.rmtree(cv_dir)

    cv_dir.mkdir(exist_ok=True, parents=True)

    pd_train_target = pd.read_csv(raw_dir / 'train_labels.csv', index_col='sample_id')
    if add_validation:
        logging.info('adding validation set to cv')
        pd_val_target = pd.read_csv(raw_dir / 'val_labels.csv', index_col='sample_id')
        pd_train_target = pd_train_target.append(pd_val_target)

    estimate_splits_to_test_size = int(np.ceil(len(pd_train_target) / test_size))
    for target_name in pd_train_target.columns:
        logger.info(f'target_name={target_name}')


        cv_iterator = RepeatedMultilabelStratifiedKFold(
            n_splits=estimate_splits_to_test_size,
            n_repeats=n_splits, random_state=cfg.RANDOM_SEED
            ).split(pd_train_target, y=pd_train_target[cfg.TARGETS].to_numpy())

        
        seen = np.zeros(len(pd_train_target), dtype=np.bool_)
        for fold, (train_idx, test_idx) in enumerate(cv_iterator):
            pd_fold = pd.DataFrame(index=pd_train_target.index, columns=['is_test'])
            pd_fold.loc[:, 'is_test'] = np.nan
            pd_fold.iloc[train_idx, 0] = False
            pd_fold.iloc[test_idx, 0] = True
            output_dir = cv_dir.joinpath(target_name)
            output_dir.mkdir(exist_ok=True, parents=True)
            output_path = output_dir / f'fold_{fold}.csv'
            logging.info(f'saving csv to {output_path}')
            pd_fold.to_csv(output_path, index=True)

            seen[test_idx] = True

        assert all(seen), 'missing test observations'

@click.command()
@click.option('--n_splits', type=int, default=cfg.N_CV_FOLDS)
def cli(n_splits: int):
    raw_dir = os.path.join(cfg.DATA_DIR, 'raw')
    val_cv_dir = os.path.join(cfg.DATA_DIR, 'cv_index', 'validation')
    final_cv_dir = os.path.join(cfg.DATA_DIR, 'cv_index', 'cv-model')
    final_test_cv_dir = os.path.join(cfg.DATA_DIR, 'cv_index', 'cv-model-test')
    write_cv_index(raw_dir, val_cv_dir, n_splits, cfg.TEST_SIZE, add_validation=False)
    write_cv_index(raw_dir, final_cv_dir, 3, cfg.FINAL_TEST_SIZE, add_validation=False)
    write_cv_index(raw_dir, final_test_cv_dir, 3, cfg.FINAL_TEST_SIZE, add_validation=True)



if __name__ == '__main__':
    setup_logging()
    cli()