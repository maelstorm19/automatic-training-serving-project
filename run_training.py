import tensorflow as tf
from model_architecture import architecture
from train import Trainer
from datasetManager.TFRecordGeneretor import TFRecordGenerator
import export_keras_to_pb
from datasetManager.TFRecordExtractor import TFRecordExtractor
import random
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
TRAIN_FILE_NAME = 'tfRecords_datasets/train_set.tfrecords'
VAL_FILE_NAME = 'tfRecords_datasets/val_set.tfrecords'
TEST_FILE_NAME = 'tfRecords_datasets/test_set.tfrecords'
MODEL_FILE_NAME = "model_repository/saved-model.h5"

##Defining callbacks

checkpointer = tf.keras.callbacks.ModelCheckpoint(MODEL_FILE_NAME, monitor='val_loss', verbose=1,
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
    generator.convert_image_folder(X_train, Y_train, "tfRecords_datasets/train_set.tfrecords")
    # Creates tfrecords from validation set
    generator.convert_image_folder(X_val, Y_val, "tfRecords_datasets/val_set.tfrecords")
    # Creating tfrecords from test set
    generator.convert_image_folder(X_test, Y_test, "tfRecords_datasets/test_set.tfrecords")

    print("train.tfrecords, val.tfrecords, test.tfrecords have successfully been created")

    # Extracting the data
    print("Creating data pipeline to feed the data to the model")
    train_dataset = TFRecordExtractor(TRAIN_FILE_NAME).extract_image(TRAIN_BATCH_SIZE)
    val_dataset = TFRecordExtractor(VAL_FILE_NAME).extract_image(TRAIN_BATCH_SIZE)
    X_test, Y_test = TFRecordExtractor(TEST_FILE_NAME).extract_image(NUM_TEST_EXAMPLES)

    # Building the model
    print("Building the model")
    resnet = architecture.VGG19Model()
    model = resnet.build_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE)

    # Training the model
    print("Training the model ...")
    trainer = Trainer()
    history = trainer.train_model(model=model, train_dataset=train_dataset,
                                  val_dataset=val_dataset, num_epochs=EPOCHS,
                                  num_train_examples=NUM_TRAIN_EXAMPLES, num_val_examples=NUM_VAL_EXAMPLES,
                                  batch_size=TRAIN_BATCH_SIZE, callbacks=callbacks)

    print("Model trained and saved at model_repository")
    print("Exporting the keras model to SavedModel format for serving with tensorflow serving")
    try:
        export_keras_to_pb.export_model()
        print("Model exported to SavedModel")
    except:
        print("Cannot export the model...")

