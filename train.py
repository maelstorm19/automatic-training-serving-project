from base import base_train
import math


class Trainer(base_train.BaseTrain):

    def train_model(self, model, **kwargs):
        '''
        :param model:
        :param kwargs:
            -   train_image, train_label : Training input features and
            training labels of type tf.data.Dataset
            -   val_image, val_label    : validation input features and
            validation labels of type tf.data.Dataset
            -   num_epochs  : Number of epochs
            -   batch_size : Batch size
            -   callbacks  : Callbacks list
            -   num_train_examples : Number of training examples
            -   num_val_examples   : Number of validation examples
        :return: Training history of type History
        '''
        train_image, train_label = kwargs.get("train_dataset")
        val_image, val_label = kwargs.get("val_dataset")

        num_epochs = kwargs.get("num_epochs")
        batch_size = kwargs.get("batch_size")
        callbacks = kwargs.get("callbacks")
        num_train_examples = kwargs.get("num_train_examples")
        num_val_examples = kwargs.get("num_val_examples")

        # Trains for num_epochs epochs
        history = model.fit(x=train_image, y=val_label, validation_data=(val_image, val_label),
                            validation_steps=math.ceil(num_val_examples / batch_size),
                            epochs=num_epochs,
                            steps_per_epoch=math.ceil(num_train_examples / batch_size),
                            callbacks=callbacks)

        return history
