import tensorflow as tf

def export_model():

    model_name = 'model_repository/saved-model.h5'
    export_path = 'serving/saved_model/1'

    new_model = tf.keras.models.load_model(model_name)
    # Export the model to a SavedModel

    tf.keras.experimental.export_saved_model(new_model, export_path)