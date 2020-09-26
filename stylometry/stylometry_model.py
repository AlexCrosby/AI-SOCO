import argparse
import sys

sys.path.append('..')
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import RMSprop

from tools.datatools import load_data, load_all_labels
from vectorizer import Vectorizer


def main(args):
    print('Settings:')
    print(str(args)[10:-1])

    length = 136
    print('Loading data...')

    if args.no_recalc:
        train_x, _, dev_x, _, _, test_x, _, _ = load_data(
            '../data_dir')  # Makes use of both raw and preprocessed source codes.
        train_x2, _, dev_x2, _, _, test_x2, _, _ = load_data('../data_dir', preprocess=True)
        print("Extracting stylometric features...")
        vec = Vectorizer(
            'lexical')  # Runs the stylometry_vectorizer from the vectorizer.py file so characters can be grabbed
        # simultaneously.
        train_x = vec.vectorize(train_x, train_x2)  # Vectorize all 3 subsets
        dev_x = vec.vectorize(dev_x, dev_x2)
        test_x = vec.vectorize(test_x, test_x2)
        del train_x2, dev_x2, test_x2
        scaler = MinMaxScaler()  # Rescale values between 0 and 1.
        print("Rescaling...")
        train_x = scaler.fit_transform(train_x)
        dev_x = scaler.transform(dev_x)
        test_x = scaler.transform(test_x)
        length = len(train_x[0])
        print(length)
        trainmm = np.memmap('vectors/train.mm', dtype='float32', mode='w+', shape=(50000, length))
        trainmm[:] = train_x[:]
        devmm = np.memmap('vectors/dev.mm', dtype='float32', mode='w+', shape=(25000, length))
        devmm[:] = dev_x[:]
        testmm = np.memmap('vectors/test.mm', dtype='float32', mode='w+', shape=(25000, length))
        testmm[:] = test_x[:]
        del trainmm, devmm, testmm, train_x, dev_x, test_x  # Save and flush all vectors.
        print("Finished building vectors.")
    # Load data from file.
    train_y, dev_y, _ = load_all_labels('../data_dir')
    dev = np.array(np.memmap('vectors/dev.mm', dtype='float32', mode='r', shape=(25000, length)))
    test = np.array(np.memmap('vectors/test.mm', dtype='float32', mode='r', shape=(25000, length)))

    train = np.array(np.memmap('vectors/train.mm', dtype='float32', mode='r', shape=(50000, length)))
    # Model.
    callback_list = [EarlyStopping(monitor='val_acc', patience=10),
                     ModelCheckpoint(filepath='style_model.h5', monitor='val_acc', save_best_only=True),
                     ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=5)]
    model = Sequential()
    model.add(Dense(500, activation='relu', input_shape=(136,)))
    model.add(Dropout(0.3))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1000, activation='softmax'))
    opt = RMSprop(learning_rate=0.001)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['acc'])
    model.summary()
    model.fit(train, train_y, epochs=1000, batch_size=250, validation_data=(dev, dev_y), shuffle=True,
              callbacks=callback_list)
    model = load_model('style_model.h5')
    print(model.evaluate(dev, dev_y))
    # Generate predictions.
    predict_vec = np.memmap('vectors/dev_style.mm', dtype='float32', mode='w+', shape=(25000, 1000))
    predict_vec[:] = model.predict(dev)[:]
    del predict_vec

    predict_vec2 = np.memmap('vectors/test_style.mm', dtype='float32', mode='w+', shape=(25000, 1000))
    predict_vec2[:] = model.predict(test)[:]
    del predict_vec2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    parser.add_argument('-r',
                        '--no_recalc',
                        help='Deactivates recalculation of vectors. Must be activated when creating new vectors.'
                             'arguments. Default = True',
                        action='store_false')
    args = parser.parse_args()
    main(args)
