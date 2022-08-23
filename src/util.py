from typing import Dict, Any, Union
import json
import logging
import coloredlogs
import sys
import os
from omegaconf import OmegaConf
from typing import Union, Dict, Any
import yaml
from pathlib import Path

pathtype = Union[str, os.PathLike]


def setup_logging(console_level=logging.INFO):
    """
    Setup logging configuration

        default_level (logging.LEVEL): default logging level
    Returns:
        None
    """
    console_format = ('%(asctime)s - %(name)s - %(levelname)-8s '
                      '[%(filename)s:%(lineno)d] %(message)s')
    root_logger = logging.getLogger()
    # Handlers for stdout/err logging
    output_handler = logging.StreamHandler(sys.stdout)
    output_handler.setFormatter(logging.Formatter(console_format))
    output_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(output_handler)
    root_logger.setLevel(logging.DEBUG)
    # setting coloredlogs
    coloredlogs.install(fmt=console_format, level=console_level,
                        sys=sys.stdout)


def allign_dictionary_subclasses(dictionary: Dict[str, Union[Any, Dict[str, Any]]]) -> Dict[str, Any]:
    output = {}
    for item, value in dictionary.items():
        if isinstance(value, dict):
            _subdict = allign_dictionary_subclasses(value)
            renamed_subdict = {f'{item}__{subitem}': subvalue for subitem, subvalue in _subdict.items()}
            output.update(renamed_subdict)
        else:
            output[f'{item}'] = value
    return output


def pretty_print_config(config: Dict[str, Any]):
    print(json.dumps(config, indent=4, sort_keys=True))


def _load_yml(config_file: pathtype):
    """Helper function to read a yaml file"""
    with open(config_file, "rb") as f:
        config = yaml.safe_load(f)
    return config


def load_config(config_file: pathtype,
                ext_vars: Dict[str, Any] = None,
                use_omega_conf: bool = False) -> Dict[str, Any]:
    """Helper function to read a config file"""

    if not os.path.exists(config_file):
        raise FileNotFoundError(f'{config_file} file do not exists')
    config = _load_yml(config_file)
    return config

def write_yml(config: Dict[str, Any], path: str) -> None:
    path = Path(path)
    parent_dir: Path = path.parents[0]
    parent_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, path)