import tensorflow as tf
import os


class TFRecordExtractor:

    def __init__(self, tfrecord_file):
        """
        :param tfrecord_file:
        """
        self.tfrecord_file = os.path.abspath(tfrecord_file)
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

    @staticmethod
    def _extract_fn(tfrecord):
        '''

        :param tfrecord:
        :return:
        '''
        # Extract features using the keys set during creation
        features = {
            'filename': tf.io.FixedLenFeature([], tf.string),
            'rows': tf.io.FixedLenFeature([], tf.int64),
            'cols': tf.io.FixedLenFeature([], tf.int64),
            'channels': tf.io.FixedLenFeature([], tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

        # Extract the data record
        sample = tf.io.parse_single_example(tfrecord, features)
        image = tf.image.decode_image(sample['image'])
        label = sample['label']
        return image, tf.one_hot(label, depth=2)

    @staticmethod
    def _train_preprocess_fn(image, label):
        '''

        :param image:
        :param label:
        :return:
        '''

        """Preprocess a single training image of layout [height, width, depth]."""
        # Resize the image to add four extra pixels on each side.
        try:
            # Randomly flip the image horizontally.
            image = tf.image.random_flip_left_right(image)
            image = tf.image.resize(image, size=[80, 80])
            image = image/255
            #image = tf.image.rgb_to_grayscale(image)

        except:

            pass

        return image, label

    def extract_image(self, batch_size):
        '''

        :param batch_size:
        :return:
        '''
        dataset = tf.data.TFRecordDataset(self.tfrecord_file)
        dataset = dataset.map(self._extract_fn, 4)
        dataset = dataset.map(self._train_preprocess_fn, 4)

        dataset = dataset.shuffle(batch_size)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(self.AUTOTUNE)
        try:
            dataset = next(iter(dataset))
        except:
            raise
        return dataset
