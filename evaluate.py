import tensorflow as tf
from datasetManager.TFRecordExtractor import TFRecordExtractor

BATCH_SIZE = 2500
Model_NAME = 'model_repository/saved-model-02-11.10.hdf5'
TEST_FILE_NAME = 'tfRecords_datasets/test_set.tfrecords'
# Recreate the exact same model purely from the file
new_model = tf.keras.models.load_model(Model_NAME)

X_test, Y_test = TFRecordExtractor(TEST_FILE_NAME).extract_image(BATCH_SIZE)

evaluation = new_model.evaluate(X_test, Y_test, steps=2)
#predictions = new_model.predict(X_test)


