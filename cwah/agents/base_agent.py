from pathlib import Path
import random
import time
import math
import copy
import pickle
import importlib
import multiprocessing

import sys
# import vh_graph
# from vh_graph.envs import belief
# from vh_graph.envs.vh_env import VhGraphEnv


class BaseAgent:
    """
    Base agent class
    """
    def __init__(self, max_episode_length):

        self.max_episode_length = max_episode_length


    def get_action(self, observation, goal):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


            
 