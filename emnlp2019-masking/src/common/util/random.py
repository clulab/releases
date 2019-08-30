import os
import random
import numpy as np
import torch

from common.training.options import gpu


class SimpleRandom():
    instance = None

    def __init__(self,seed):
        self.seed = seed
        self.random = random.Random(seed)

    def next_rand(self,a,b):
        return self.random.randint(a,b)

    @staticmethod
    def get_instance():
        if SimpleRandom.instance is None:
            SimpleRandom.instance = SimpleRandom(SimpleRandom.get_seed())
        return SimpleRandom.instance

    @staticmethod
    def get_seed():
        return int(os.getenv("RANDOM_SEED", 12459))

    @staticmethod
    def get_seed_from_config_file(seed_from_config_file):
        return int(seed_from_config_file)

    @staticmethod
    def set_seeds():

        torch.manual_seed(SimpleRandom.get_seed(seed_from_config_file))
        if gpu():
            torch.cuda.manual_seed_all(SimpleRandom.get_seed())
        np.random.seed(SimpleRandom.get_seed())
        random.seed(SimpleRandom.get_seed())

    @staticmethod
    def set_seeds_from_config_file(seed_from_config_file):
        torch.manual_seed(SimpleRandom.get_seed_from_config_file(seed_from_config_file))
        if gpu():
            torch.cuda.manual_seed_all(SimpleRandom.get_seed_from_config_file(seed_from_config_file))
        np.random.seed(SimpleRandom.get_seed_from_config_file(seed_from_config_file))
        random.seed(SimpleRandom.get_seed_from_config_file(seed_from_config_file))