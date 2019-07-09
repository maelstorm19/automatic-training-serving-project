import tensorflow as tf
from datasetManager.TFRecordExtractor import TFRecordExtractor
import argparse

BATCH_SIZE = 2500
Model_NAME = 'model_repository/saved-model.h5'
TEST_FILE_NAME = 'tfRecords_datasets/test_set.tfrecords'
# Recreate the exact same model purely from the file
ap = argparse.ArgumentParser()

ap.add_argument('-num_samples', default=BATCH_SIZE, type=int, help="Number of test samples")

args = vars(ap.parse_args())

print("Proceeding to the evaluation of the model on the test set...")

new_model = tf.keras.models.load_model(Model_NAME)

X_test, Y_test = TFRecordExtractor(TEST_FILE_NAME).extract_image(args['num_samples'])

scores= new_model.evaluate(X_test, Y_test, steps=2)

print("%s: %.2f%%" % (new_model.metrics_names[1], scores[1] * 100))
print("Model evaluation finished.")

