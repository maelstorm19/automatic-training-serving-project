FROM python:3.6.5

RUN apt-get update

RUN apt-get install sudo -y

RUN apt-get install curl -y
RUN apt-get install nano -y
RUN apt-get install git -y

#Downloading tensorflow-serving repo
RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

RUN apt-get update


#Installing Tensorflow-model-server
RUN apt-get install tensorflow-model-server -y



WORKDIR /serving_app

COPY . /serving_app

RUN mkdir tfRecords_datasets

RUN mkdir logs

RUN mkdir model_repository

RUN mkdir serving_models


RUN pip install -r requirements.txt



