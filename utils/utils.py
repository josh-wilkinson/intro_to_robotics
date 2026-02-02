
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

def difference_angle(angle1, angle2, is_degrees=False):
    """
    Computes the difference between two angles in radians,
    taking into account the circular nature of angles.
    The result is in the range [-pi, pi].
    """
    if is_degrees:
        angle1 = np.deg2rad(angle1)
        angle2 = np.deg2rad(angle2)
    diff = angle2 - angle1
    while diff < -np.pi:
        diff += 2 * np.pi
    while diff > np.pi:
        diff -= 2 * np.pi
    return diff

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