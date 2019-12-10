import yaml
import os
import logging
import importlib
from collections import namedtuple


def dict_to_namedtuple(typename, dictionary):
    for k in dictionary:
        if isinstance(dictionary[k], dict):
            dictionary[k] = dict_to_namedtuple(str(k) + "_", dictionary[k])
    return namedtuple(typename, dictionary.keys())(**dictionary)


def load_module_from_source(src_path):
    loader = importlib.machinery.SourceFileLoader('nnModel', src_path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def parse_yaml_config(yaml_file_path):
    """
    Load a YAML configuration file.
    :type yaml_file_path: str
    Examples:
    >>> config = parse_yaml_config("../experiments/config.yaml")
    >>> _ = config.system.log_dir

    """
    # Read YAML experiment definition file
    with open(yaml_file_path, 'r', encoding='utf-8') as stream:
        cfg = yaml.load(stream)
    cfg = make_paths_absolute(os.path.dirname(yaml_file_path), cfg)
    cfg = dict_to_namedtuple("ConfigType", cfg)
    return cfg


def make_paths_absolute(dir_, cfg):
    """
    Make all values for keys ending with `_path` absolute to dir_.
    """
    for key in cfg.keys():
        if key.endswith("_path") or key.endswith("_dir"):
            cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
            if not os.path.exists(cfg[key]):
                logging.warning("%s does not exist.", cfg[key])
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    return cfg


if __name__ == '__main__':
    import doctest

    doctest.testmod()