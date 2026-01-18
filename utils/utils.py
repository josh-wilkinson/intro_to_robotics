
import os
import time
import numpy as np

import yaml
from rich import print
from enum import Enum


class boundary(Enum):
    # aruco marker modulo 3:
    obstacle = 1
    inner = 0
    outer = 2


def load_config(filepath="./config.yaml"):
    if os.path.isfile(filepath):
        with open(filepath, "r") as stream:
            try:

                # param_list = rosparam.load_file(filepath)
                # print("param_list", param_list)

                yaml_parsed = yaml.safe_load(stream)

                print("config:", yaml_parsed)

                return Struct(yaml_parsed)
            except yaml.YAMLError as exc:
                print("yaml error", exc)
    else:
        print(f"{filepath} is not a file!")
    return None


class Struct(object):
    """
    Holds the configuration for anything you want it to.
    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    """

    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return Struct(value) if isinstance(value, dict) else value

    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)