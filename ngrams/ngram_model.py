import argparse
import sys

import numpy as np
from joblib import dump
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

from ngrams.ngram_vectorizer import NgramVectorizer

sys.path.append('..')
from tools.datatools import load_data, plot_history, write_predictions
from tools.generators import GeneratorX, Generator


def main(args):
    print('Settings:')
    print(str(args)[10:-1])

    transform_size = 1000  # Batch size to be used during transformation.

    num_features = args.features
    print('Loading data')
    train_x, train_y, dev_x, dev_y, dev_pid, test_x, test_y, test_pid = load_data(args.data, bytes=args.byte,
                                                                                  preprocess=args.preprocessed)

    if args.no_recalc:  # Fit vectorizer and transform source codes to n-gram bag vectors.
        print('Calculating vectors.')
        vec = train_vectorizer(train_x, args.mode, args.ngram, num_features)
        dump(vec, 'vectorizer.joblib')  # Saves the vectorizer to be used in n-gram error analysis
        batch_transform(vec, train_x, 'train', num_features, transform_size)
        batch_transform(vec, dev_x, 'dev', num_features, transform_size)
        batch_transform(vec, test_x, 'test', num_features, transform_size)
        if args.mode == 'c':
            print("Rescaling count values.")
            rescale(num_features)
    else:
        print('Vector calculation skipped, loading from pre-calculated files.')
    predictions_dev, predictions_test, history = run_model(args.batch, num_features, train_y, dev_y, args.skip,
                                                           args.fullpredict)
    if not args.skip:
        plot_history(history)  # Plot training and validation accuracy and loss per epoch.
    if args.results:
        write_predictions(predictions_dev, dev_pid, 'dev_predictions')
        write_predictions(predictions_test, test_pid, 'test_predictions')


def train_vectorizer(train_x, mode, n_size, num_features):  # Trains n-gram vectorizer
    if mode not in 'bct':
        raise Exception('Invalid mode - please pick \'b\', \'c\' or \'t\'.')
    vec = NgramVectorizer(size=n_size, matrix=True, count_mode=mode, max_features=num_features)
    vec.fit(train_x)
    return vec


def batch_transform(vectorizer, data, file_name, num_features, fit_size):  # Transforms source codes to vectors.
    # Due to the size of vectors, they are stored in a memmap object as to not exceed ram limit.
    print(f'Fitting {file_name}.')
    fp = np.memmap(f'vectors/{file_name}.mm', dtype='float32', mode='w+', shape=(len(data), num_features))
    x_generator = GeneratorX(data, fit_size)
    for i in range(x_generator.__len__()):
        x = vectorizer.transform(x_generator.__getitem__(i))
        fp[i * fit_size:(i + 1) * 1000] = x
    del fp


def rescale(num_features):  # MinMax scales values when count features are used.
    scaler = MinMaxScaler()
    train = np.memmap('vectors/train.mm', dtype='float32', mode='r+', shape=(50000, num_features))
    dev = np.memmap('vectors/dev.mm', dtype='float32', mode='r+', shape=(25000, num_features))
    test = np.memmap('vectors/test.mm', dtype='float32', mode='r+', shape=(25000, num_features))
    train[:] = scaler.fit_transform(train)
    dev[:] = scaler.transform(dev)
    test[:] = scaler.transform(test)
    del train, dev, test


def run_model(batch_size, num_features, train_y, dev_y, skip_training=False):
    # Loads generators ands runs the model
    train = np.memmap('vectors/train.mm', dtype='float32', mode='r', shape=(50000, num_features))
    dev = np.memmap('vectors/dev.mm', dtype='float32', mode='r', shape=(25000, num_features))

    test = np.memmap('vectors/test.mm', dtype='float32', mode='r', shape=(25000, num_features))

    train = Generator(train, train_y, batch_size)
    dev_p = GeneratorX(dev, batch_size)
    dev = Generator(dev, dev_y, batch_size)
    test = GeneratorX(test, batch_size)

    callback_list = [EarlyStopping(monitor='val_acc', patience=10),
                     ModelCheckpoint(filepath='ngram_model.h5', monitor='val_acc', save_best_only=True),
                     ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=5)]
    ####################################################################################################################
    # The model.
    model = Sequential()
    model.add(Dense(3000, activation='relu', input_shape=(num_features,)))
    model.add(Dropout(0.5))
    model.add(Dense(2000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    opt = RMSprop(learning_rate=0.0001)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['acc'])
    model.summary()
    ####################################################################################################################
    if not skip_training:
        history = model.fit(train, epochs=1000, validation_data=dev, callbacks=callback_list, shuffle=True)
    else:
        history = None
    model.load_weights('ngram_model.h5')
    score = model.evaluate(dev)
    print(score)
    # Generates outputs for ensemble.
    predict_vec = np.memmap('vectors/dev_ngram.mm', dtype='float32', mode='w+', shape=(25000, 1000))
    predict_vec[:] = model.predict(dev_p)[:]
    del predict_vec
    predict_vec2 = np.memmap('vectors/test_ngram.mm', dtype='float32', mode='w+', shape=(25000, 1000))
    predict_vec2[:] = model.predict(test)[:]
    del predict_vec2
    predictions_dev = model.predict(dev_p).argmax(axis=-1)
    predictions_test = model.predict(test).argmax(axis=-1)
    return predictions_dev, predictions_test, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('-d',
                        '--data',
                        type=str,
                        default=r'../data_dir',
                        help='Path to the data directory.')
    parser.add_argument('-p',
                        '--preprocessed',
                        help='Whether to use preprocessed data or not. Default = false.',
                        action='store_true')
    parser.add_argument('-m',
                        '--mode',
                        type=str,
                        default='b',
                        help='Mode for the vectorizer: Options are b (binary), c (count), t (TF-IDF). Default = b',
                        )
    parser.add_argument('-n',
                        '--ngram',
                        type=int,
                        default=6,
                        help='N-gram size. Default = 6'
                        )
    parser.add_argument('-f',
                        '--features',
                        type=int,
                        default=20000,
                        help='The number of features to use. Default = 20000',
                        )
    parser.add_argument('-b',
                        '--batch',
                        type=int,
                        default=32,
                        help='The batch size to use in the neural net. Default = 32',
                        )
    parser.add_argument('-r',
                        '--no_recalc',
                        help='Deactivates recalculation of vectors. Must be activated when creating new vectors.'
                             'arguments. Default = True',
                        action='store_false')
    parser.add_argument('-y',
                        '--byte',
                        help='Whether to load the data files as bytes (True) or as strings (False). Default = False',
                        action='store_true')
    parser.add_argument('-s',
                        '--skip',
                        help='Whether to skip training and just load model from file. Default = False',
                        action='store_true')
    parser.add_argument('-u',
                        '--results',
                        help='Whether to write dev and test final predictions to file for submission. Default = False',
                        action='store_true')
    args = parser.parse_args()
    main(args)
