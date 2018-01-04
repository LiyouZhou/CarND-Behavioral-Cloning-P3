#!/usr/bin/env python
import csv, cv2, pickle
import numpy as np
from os import path
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def data_generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                img = cv2.imread(batch_sample[0])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if batch_sample[1]:
                    img = np.fliplr(img)

                angle = float(batch_sample[2])

                images.append(img)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def main(data_paths, output_fn, correction, nb_epoch, base_model, save_hist):
    # prepare data
    samples = []
    for data_path in data_paths:
        data_csv_path = path.join(data_path, "driving_log.csv")
        with open(data_csv_path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                for i, img_fn in enumerate(line[:3]):
                    img_fn = path.basename(img_fn)
                    img_fn = path.join(data_path, "IMG", img_fn)

                    measurement = float(line[3])
                    if   ('left' in img_fn): measurement += correction
                    elif ('right' in img_fn): measurement -= correction

                    # path to image, flipped?, steering angle
                    samples.append((img_fn, False, measurement))
                    samples.append((img_fn, True, -measurement))

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    train_generator = data_generator(train_samples, batch_size=32)
    validation_generator = data_generator(validation_samples, batch_size=32)

    # define model
    dropput_rate = 0.5
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
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
    model.add(Dense(100))
    model.add(Dropout(dropput_rate))
    model.add(Dense(50))
    model.add(Dropout(dropput_rate))
    model.add(Dense(10))
    model.add(Dense(1))

    # train model
    model.compile(loss="mse", optimizer="adam")
    model.summary()

    # load a already trained model if specified
    if base_model:
        model.load_weights(base_model)

    history_object = model.fit_generator(train_generator,
                                         samples_per_epoch=len(train_samples),
                                         validation_data=validation_generator,
                                         nb_val_samples=len(validation_samples),
                                         nb_epoch=nb_epoch)

    # save output
    model.save(output_fn)

    # save the training history to file
    if save_hist:
        with open(output_fn.split(".")[0] + "_traing_hist.p", 'wb') as file_pi:
            pickle.dump(history_object.history, file_pi)

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
    parser.add_argument("-s", "--save-history", action="store_true", default=False,
        help="File where to save the training history")
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
         base_model = args.base_model,
         save_hist  = args.save_history)
