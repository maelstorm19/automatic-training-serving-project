import tensorflow as tf

def export_model(export_path,model_path_name):
    '''
    Export h5 model to SavedModel protocol buffer format for serving
    :return: None
    '''

    new_model = tf.keras.models.load_model(model_path_name)
    # Export the model to a SavedModel
    tf.keras.experimental.export_saved_model(new_model, export_path)