from base import base_model
import tensorflow as tf
from tensorflow.python.keras.applications import VGG19, ResNet50
from tensorflow.python.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Sequential


'''Create a class for youyr model and build the architecture'''
class Resnet50Model(base_model.BaseModel):


    def build_model(self, input_shape, **kwargs):

        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1000)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.nn.relu)(x)
        x = tf.keras.layers.Dense(1000)(x)
        x = tf.keras.layers.Activation(tf.nn.relu)(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        predictions = tf.keras.layers.Dense(kwargs.get("num_classes"), activation=tf.nn.softmax)(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
        print(len(base_model.layers))
        # i.e. freeze all convolutional Resnet layers
        for layer in base_model.layers:
            layer.trainable = False

            # The compile step specifies the training configuration.
            if kwargs.get("learning_rate") is not None:
                lr = kwargs.get("learning_rate")
            else:
                lr = 0.001
            model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print(model.summary())
        return model


class VGG19Model(base_model.BaseModel):

    def build_model(self, input_shape, **kwargs):


        inputs = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(64,(3, 3))(inputs)
        x = tf.keras.layers.MaxPool2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(64, (3, 3))(x)
        x = tf.keras.layers.MaxPool2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(64, (3, 3))(x)
        x = tf.keras.layers.MaxPool2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(64, (3, 3))(x)
        x = tf.keras.layers.MaxPool2D((2, 2))(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(100)(x)

        predictions = tf.keras.layers.Dense(kwargs.get("num_classes"), activation=tf.nn.softmax)(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=predictions)

        if kwargs.get("learning_rate") is not None:
            lr = kwargs.get("learning_rate")
        else:
            lr = 0.001

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print(model.summary())
        return model


class HeuritechModel(base_model.BaseModel):

    def build_model(self, input_shape, **kwargs):
        model = Sequential()
        # Adds a densely-connected layer with 64 units to the model:
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Add another:
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        # Add a softmax layer with 10 output units:
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer="adam",
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        print(model.summary())

        return model

