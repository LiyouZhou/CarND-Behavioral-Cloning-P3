#!/usr/bin/env python
import pickle
import matplotlib.pyplot as plt

def main(data):
    # plot the training and validation loss for each epoch
    plt.plot(data['loss'])
    plt.plot(data['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')

    plt.show()
    # plt.imsave(output_fn + "training_losses.png")

if __name__ == '__main__':
    import argparse
    from glob import glob

    parser = argparse.ArgumentParser(
        description="plot training losses")
    parser.add_argument("-i", "--input-file", type=argparse.FileType('rb'), required=True,
        help="input pickle file")
    args = parser.parse_args()

    hist = pickle.load(args.input_file)

    main(hist)