#!/usr/bin/env python

import csv, cv2
import numpy as np
from os import path
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

def main(data_paths, output_fn, correction, nb_epoch, base_model):
    # prepare data
    images = []
    measurements = []
    for data_path in data_paths:
        data_csv_path = path.join(data_path, "driving_log.csv")
        with open(data_csv_path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                for i, img_fn in enumerate(line[:3]):
                    img_fn = path.basename(img_fn)
                    img_fn = path.join(data_path, "IMG", img_fn)

                    img = cv2.imread(img_fn)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    images.append(np.fliplr(img))

                    measurement = float(line[3])
                    if   ('left' in img_fn): measurement += correction
                    elif ('right' in img_fn): measurement -= correction
                    measurements.append(measurement)
                    measurements.append(-measurement)

    X_train = np.array(images)
    y_train = np.array(measurements)

    print(X_train.shape)

    # define model
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=X_train.shape[1:]))
    model.add(Lambda(lambda x: x/255.0 - 0.5))
    model.add(Convolution2D(24, 5, 5, activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(36, 5, 5, activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(48, 5, 5, activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    # train model
    model.compile(loss="mse", optimizer="adam")

    # load a already trained model if specified
    if base_model:
        model.load_weights(base_model)

    history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=nb_epoch)

    # save output
    model.save(output_fn)

    # print the keys contained in the history object
    print(history_object.history.keys())

    # plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.imsave("training_losses.png")

if __name__ == '__main__':
    import argparse
    from glob import glob

    parser = argparse.ArgumentParser(
        description="Train a model to drive a simulated vehicle around a simulated track")
    parser.add_argument("-o", "--output", type=str, required=True, default="model.h5",
        help="output model file name")
    parser.add_argument("-i", "--input-folder", type=str, required=True, nargs ='*',
        help="input folders, You can also specify multiple folders: -i <path1> <path2>'")
    parser.add_argument("-b", "--base-model", type=str, default=None,
        help="load a trained model to start with")
    parser.add_argument("-n", "--nb-epoch", type=int,
        help="number of epoch", default=3)
    parser.add_argument("-c", "--correction", type=float, default=0.2,
        help="correction to the middle")
    args = parser.parse_args()

    print(args)

    data_paths = []
    for p in args.input_folder:
        data_paths += glob(p)
    print(data_paths)

    main(data_paths = data_paths,
         output_fn  = args.output,
         correction = args.correction,
         nb_epoch   = args.nb_epoch,
         base_model = args.base_model)
