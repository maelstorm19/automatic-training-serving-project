
import pathlib
import numpy
import tensorflow as tf
import os
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import numpy as np



class TFRecordGenerator:

    '''
        Generates generate and save images as tfrecords files
    '''
    def __init__(self, path_root):
        self.path_root = path_root
        self.data_root = pathlib.Path(self.path_root)

    def convert_image_folder(self, all_image_paths, all_image_labels, tfrecord_file_name):
        '''
        :param all_image_paths:
        :param all_image_labels:
        :param tfrecord_file_name:
        :return:
        '''
        with tf.io.TFRecordWriter(tfrecord_file_name) as writer:

            for i, img_path in enumerate(all_image_paths):
                    example = self._convert_image(img_path,all_image_labels, i)
                    writer.write(example.SerializeToString())
                    #print("All the images present in the given have successfully been converted to TFRecord files")

    @staticmethod
    def _convert_image(img_path, all_image_labels, img_path_index):
        '''
        :param img_path:
        :param all_image_labels:
        :param img_path_index:
        :return: None
        '''
        label = all_image_labels[img_path_index]
        img_shape = mpimg.imread(img_path).shape
        filename = os.path.basename(img_path)

        # Read image data in terms of bytes
        with tf.io.gfile.GFile(img_path, 'rb') as fid:
            image_data = fid.read()



        example = tf.train.Example(features=tf.train.Features(feature={
            'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf-8')])),
            'rows': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_shape[0]])),
            'cols': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_shape[1]])),
            'channels': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_shape[2]])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        }))
        return example

    def get_all_image_paths(self, files_depth):
        '''
        :param files_depth: depth for which the desired files can be accessed in the form */, */* etc..
        :return: paths of all images in your directories
        '''

        all_image_paths = list(self.data_root.rglob(files_depth))
        all_image_paths = [str(path) for path in all_image_paths]
        #random.shuffle(all_image_paths)

        image_count = len(all_image_paths)
        print("your dataset contains "+  str(image_count) + " images")

        return all_image_paths

    def label_builder(self, all_image_paths):
        '''

        :param all_image_paths: paths of all the images in your dataset directory
        :return: the labels of all the images
        '''
        label_names = sorted(item.name for item in self.data_root.glob('*/') if item.is_dir())
        label_to_index = dict((name, index) for index, name in enumerate(label_names))

        all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

        return all_image_labels

    def heuritech_label_builder(self, all_image_paths):
        '''
        This function is a generic function of the create_test_data function provided
        it automatically creates unique labels based on the label names
        :param all_image_paths:
        :return:
        '''
        label_names = sorted(name.split('.')[0].split('/')[2] for name in all_image_paths)
        label_to_index = dict((name, index) for index, name in enumerate(np.unique(label_names)))
        all_image_labels = [label_to_index[path.split('.')[0].split('/')[2]] for path in all_image_paths]

        return all_image_labels


    def train_test_validation_split(self, X, Y, set_weights=(8, 1, 1),
                                    print_sets_size=True, stratify=False):
        """
        Create three sets --> train, test and validation

        Here we extend the sklearn.model_selection.train_test_split function to
        output three datasets !
        The weights will be then uniformation to sum up to 1.
        :param X: Observation Dataframe
        :param Y: Classes Dataframe
        :param set_weights: set of three elements corresponding to (train, test, val)
        :param print_sets_size: Boolean indicating if we print the set sizes
        :param stratify: Boolean indicating if we stratigy or not
        :return: X_train, X_test, X_val, Y_train, Y_test, Y_val
        """
        w_train, w_test, w_val = set_weights

        # Create train sets
        train_size = w_train / sum(set_weights)
        y_stratify = None if stratify is False else Y
        X_train, X_tmp, Y_train, Y__tmp = train_test_split(X, Y,
                                                           train_size=train_size,
                                                           stratify=y_stratify)

        # Create test and validation sets
        test_size = w_test / (w_test + w_val)
        y_stratify = None if stratify is False else Y__tmp
        X_test, X_val, Y_test, Y_val = train_test_split(X_tmp, Y__tmp,
                                                        train_size=test_size,
                                                        stratify=y_stratify)

        if print_sets_size is True:
            print("Dataset train size:", numpy.shape(X_train))
            print("Dataset test size:", numpy.shape(X_test))
            print("Dataset validation size:", numpy.shape(X_val))

        return X_train, X_test, X_val, Y_train, Y_test, Y_val


