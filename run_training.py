import tensorflow as tf
from model_architecture import architecture
from train import Trainer
from datasetManager.TFRecordGeneretor import TFRecordGenerator
import export_keras_to_pb
from datasetManager.TFRecordExtractor import TFRecordExtractor
import random
import argparse

random.seed(10)

'''
Defining model paremeters and hyperparemeters 
'''
# Defining input
INPUT_SHAPE = (80, 80, 3)  # Input shape (HEIGHT, WIDTH, CHANNELS)


TRAIN_IMAGES_DIR = 'dataset_repository/train/'
LEARNING_RATE = 3e-3
EPOCHS = 2
VAL_BATCH_SIZE = 32
TRAIN_BATCH_SIZE = 32
NUM_TRAIN_EXAMPLES = 20000
NUM_VAL_EXAMPLES = 2500
NUM_TEST_EXAMPLES = 2500
NUM_CLASSES = 2
TRAIN_FILE_PATH = 'tfRecords_datasets/train_set.tfrecords'
VAL_FILE_PATH = 'tfRecords_datasets/val_set.tfrecords'
TEST_FILE_PATH = 'tfRecords_datasets/test_set.tfrecords'
MODEL_FILE_PATH= "model_repository/saved-model.h5"
EXPORT_DIR = 'serving_models/'
MODEL_VERSION = '1'

ap = argparse.ArgumentParser()

ap.add_argument('-lr', default=LEARNING_RATE, help="Learning rate", type=float)
ap.add_argument('-epochs', default=EPOCHS, help="Number of epochs", type=int)
ap.add_argument('-batch_size', default=TRAIN_BATCH_SIZE, help="Batch size", type=int)
ap.add_argument('-num_classes', default=NUM_CLASSES, help="Number of classes", type=int)
ap.add_argument('-val_examples', default=NUM_VAL_EXAMPLES, help='Validation set size', required=False, type=int )
ap.add_argument('-train_examples', default=NUM_TRAIN_EXAMPLES, help="Training set size", required=False, type=int)
ap.add_argument('-test_examples', default=NUM_TEST_EXAMPLES, help="Training set size", required=False, type=int)

ap.add_argument('-train_file_path', default=TRAIN_FILE_PATH, help="Path to save the training set in .tfrecords format", required=False)
ap.add_argument('-val_file_path', default=VAL_FILE_PATH, help="Path to save the validation set in .tfrecords format", required=False)
ap.add_argument('-test_file_path', default=TEST_FILE_PATH, help="Path to save the test set in .tfrecords format", required=False)

ap.add_argument('-Training_images_dir', default=TRAIN_IMAGES_DIR, help="Training set directory", required=False)
ap.add_argument('-model_path_name', default=MODEL_FILE_PATH, help="Path to save the .h5 model", required=False)
ap.add_argument('-model_version', default=MODEL_VERSION, help="Version of the SavedModel, default is 1", required=False)


args = vars(ap.parse_args())

##Defining callbacks

checkpointer = tf.keras.callbacks.ModelCheckpoint(MODEL_FILE_PATH, monitor='val_loss', verbose=1,
                                                  save_best_only=False, save_weights_only=False,
                                                  mode='auto', period=2)


early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20,
                                                 verbose=1, mode='auto', baseline=None,
                                                 restore_best_weights=False)

tensor_bord=tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
                                           write_images=False, embeddings_freq=0,
                                           embeddings_layer_names=None, embeddings_metadata=None,
                                           embeddings_data=None, update_freq='epoch')

#reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                # patience=5, min_lr=LEARNING_RATE)

callbacks = [checkpointer, early_stopper, tensor_bord]

if __name__ == '__main__':


    # Generating tfrecords files from images in dataset_repository
    print("Started to create tfrecords files from dataset_repository")

    generator = TFRecordGenerator(TRAIN_IMAGES_DIR)

    all_training_set = generator.get_all_image_paths("*/")
    all_training_labels = generator.heuritech_label_builder(all_training_set)

    # Creating train, val test splits
    X_train, X_test, X_val, Y_train, Y_test, Y_val = generator.train_test_validation_split(all_training_set,
                                                                                                 all_training_labels)
    # Creating tfrecords for training set
    generator.convert_image_folder(X_train, Y_train, args['train_file_path'])
    # Creates tfrecords from validation set
    generator.convert_image_folder(X_val, Y_val, args['val_file_path'])
    # Creating tfrecords from test set
    generator.convert_image_folder(X_test, Y_test,  args['test_file_path'])

    print(args['train_file_path'],
          args['val_file_path'],
          args['val_file_path'],
          "have successfully been created")

    # Extracting the data
    print("Creating data pipeline to feed the data to the model")
    train_dataset = TFRecordExtractor(TRAIN_FILE_PATH).extract_image(args['batch_size'])
    val_dataset = TFRecordExtractor(VAL_FILE_PATH).extract_image(args['batch_size'])
    X_test, Y_test = TFRecordExtractor(TEST_FILE_PATH).extract_image(args['test_examples'])

    # Building the model
    print("Building the model")
    my_model = architecture.MyModel()
    model = my_model.build_model(input_shape=INPUT_SHAPE, num_classes=args['num_classes'], learning_rate=args['lr'])

    # Training the model
    print("Training the model ...")
    trainer = Trainer()
    history = trainer.train_model(model=model, train_dataset=train_dataset,
                                  val_dataset=val_dataset, num_epochs=args['epochs'],
                                  num_train_examples=args['train_examples'], num_val_examples=args['val_examples'],
                                  batch_size=args['batch_size'], callbacks=callbacks)


    print("Model trained and saved at model_repository")


    print("Exporting the keras model to SavedModel into serving directory for serving with tensorflow serving")
    try:
        export_keras_to_pb.export_model(EXPORT_DIR + args['model_version'], args['model_path_name'])
        print("Model exported to SavedModel")
    except:
        print("Cannot export the model...")

