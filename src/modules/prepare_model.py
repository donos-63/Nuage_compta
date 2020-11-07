# USAGE
# python train_ocr_model.py --az a_z_handwritten_data.csv --model handwriting.model

import os
import sys
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

src_dir = os.path.join(os.getcwd(), 'src')
sys.path.insert(0,src_dir) 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import the necessary packages
from tensorflow.python.keras import models
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import cv2
from modules.tools.data_helper import load_mnist_dataset
from modules.tools.data_helper import load_az_dataset
from modules.models.resnet import ResNet
import modules.tools.file_helper as file_help
from tensorflow.python.keras.callbacks import CSVLogger

def prepare_model(dataset_path, output_path):
    # initialize the number of epochs to train for, initial learning rate,
    # and batch size
    EPOCHS = 10 #50
    INIT_LR = 1e-1
    BATCH_SIZE = 32 #128

    # load the A-Z and MNIST datasets, respectively
    print("[INFO] loading datasets...")
    (data, labels) = load_az_dataset(dataset_path)

    tmp = np.array(labels) 
    labels_proceed = np.unique(tmp)

    # each image in the A-Z and MNIST digts datasets are 28x28 pixels;
    # however, the architecture we're using is designed for 32x32 images,
    # so we need to resize them to 32x32
    #data = [cv2.resize(image, (32, 32)) for image in data]
    #data = np.array(data, dtype="float32")

    # add a channel dimension to every image in the dataset and scale the
    # pixel intensities of the images from [0, 255] down to [0, 1]
    data = np.expand_dims(data, axis=-1)
    data /= 255.0

    # convert the labels from integers to vectors
    le = LabelBinarizer()
    labels = le.fit_transform(labels)
    counts = labels.sum(axis=0)

    # account for skew in the labeled data
    classTotals = labels.sum(axis=0)
    classWeight = {}

    # loop over all classes and calculate the class weight
    for i in range(0, len(classTotals)):
        classWeight[i] = classTotals.max() / classTotals[i]

    # partition the data into training and testing splits using 80% of
    # the data for training and the remaining 20% for testing
    (trainX, testX, trainY, testY) = train_test_split(data,
        labels, test_size=0.20, stratify=labels, random_state=42)

    # construct the image generator for data augmentation
    aug = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.05,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        horizontal_flip=False,
        fill_mode="nearest")

    # define the list of label names
    labelNames = "0123456789"
    labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labelNames = [l for l in labelNames]
    labelNames = [labelNames[i] for i in labels_proceed]

    # initialize and compile our deep neural network
    # print("[INFO] compiling model...")
    opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model = ResNet.build(28, 28, 1, len(le.classes_), (3, 3, 3),
            (64, 64, 128, 256), reg=0.0005)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
            metrics=["accuracy"])

    if os.path.exists(os.path.join(output_path, file_help.DEFAULT_MODEL_NAME)):
        print("model already computed in "+ os.path.join(output_path, file_help.DEFAULT_MODEL_NAME))
        model = models.load_model(os.path.join(output_path, file_help.DEFAULT_MODEL_NAME))
        histo = file_help.load_training_history()
    else:
        # train the network
        print("[INFO] training network...")
        csv_logger = CSVLogger(os.path.join(file_help.DATA_OUT_FOLDER, file_help.MODELS_FOLDER, file_help.DEFAULT_HISTORY_NAME), append=False, separator=';')
        H = model.fit(
                aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
                validation_data=(testX, testY),
                steps_per_epoch=BATCH_SIZE,
                epochs=EPOCHS,
                class_weight=classWeight,
                verbose=1,
                callbacks=[csv_logger])

        # save the model to disk
        print("[INFO] serializing network...")
        model.save(os.path.join(output_path, file_help.DEFAULT_MODEL_NAME))

        # construct a plot that plots and saves the training history
        # evaluate the network
        print("[INFO] evaluating network...")
        predictions = model.predict(testX, batch_size=BATCH_SIZE)
        print(classification_report(testY.argmax(axis=1),
        predictions.argmax(axis=1), target_names=labelNames))
        
        N = np.arange(0, EPOCHS)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, H.history["loss"], label="train_loss")
        plt.plot(N, H.history["val_loss"], label="val_loss")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(output_path, file_help.DEFAULT_PLOT_NAME))

if __name__ == "__main__":
    #test with Z
    prepare_model('C:\\prairie\\projet9\\Nuage_compta\\data\\curated\\handwritten_data.csv', 'C:\\prairie\\projet9\\Nuage_compta\\data\\out\\models')
