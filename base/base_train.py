import tensorflow as tf
from tensorflow.python import keras
import numpy as np


class BaseTrain:

    def train_model(self, unused_training_input, **unused_model_param):
        """

        :param unused_training_input: Training input like the model architecture
        :param unused_model_param: Train/val datasets and all useful necessary hyperparemeters to train the model
        :return:
        """

        raise NotImplementedError


