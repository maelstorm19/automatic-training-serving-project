import tensorflow as tf


class BaseModel:
    def __init__(self):

        print("instanciating the model")

    def build_model(self, unused_model_input, **unused_model_param):
        '''

        :param unused_model_input: Model input like input_shape
        :param unused_model_param: All useful hyperparameters necessary to build the architecture like learning rate, optimizer, etc
        :return:
        '''

        raise NotImplementedError()

