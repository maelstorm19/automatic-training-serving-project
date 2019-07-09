# automatic-training-serving-project
This repository is a project to automatically train and serve a classifier model

## part1. Introduction

After following all the steps mentionned below, you will be able to train and serve a cat vs dog classifier in no time!
Most of all you'll be able to use this repository as a template for your classification tasks.
I tried as much as possible to make the code in this repository as cleaner and understandable as possible.
So if you think we can make it even more usable and cleaner, let me know, I will be glad to here!

## part2. Project Structure

- base :  Contains the base classes for building and training models
- dataset_repository :  Contains the raw training images
- datasetManager :  A custom module to deal with the dataset and applying some ETL before training
- logs : Training process logs are stored there
- model_architecture/architecture.py :
        This module contains classes which are the names of the model you want to build.
        All the classes should implement the method build_model from the base class.

model_repository: Contains the .h5 model after the training

- export_keras_to_pb.py : A script to export the keras model to the SavedModel format for tensorflow serving
- make_inference.py: A script to test the newly created model on your images and test the serving endpoint

- run_training : A script to train the model
- evaluate.py : E
- serving: Contains the different versions of the exported SavedModel models
- tfRecords_datasets : Contains the generated train, test, val sets in the .tfrecords format


## part3. Project prerequisites

To avoid problems caused by different os(Mac, PC) and dependencies, I mounted a docker images to run the project.

So make sure you have docker installed on your machine before.


## part4. Quick start

- Clone this repository : git clone https://github.com/maelstorm19/automatic-training-serving-project.git
- Go to the the cloned repository and run :  mkdir dataset_repository
- Download the cat vs dog training(train and test1 files images on kaggle :  https://www.kaggle.com/ruchibahl18/cats-vs-dogs-basic-cnn-tutorial
- Unzip the downloaded files into the dataset_repository directory(both train and test1 directory shoulld be there)
- Move to the same directory as the Dockerfile and build the image : docker build -t cat-vs-dog-train-serv .
- Launch the docker image with port 9000 open : docker run -it -t -p 9000:9000 cat-vs-dog-train-serv /bin/bash
- inside the docker image, you can :

    - train a model by running the command :  python run_training.py -lr={'Your learning rate'} -batch_size={'Your batch_size'} -epochs={'number of epochs'}
    - serve a model by running the command : tensorflow_model_server --model_base_path=/serving_app/serving/saved_model --rest_api_port=9000 --model_name=cat-dog
    - Evaluate the accuracy of the model by running : python evaluate.py -num_samples={'Number of test samples'}
You can now test the served model outside of the container by running the command: python make_inference --image={'Your image path'}

## Copyright

Ismael Goulani