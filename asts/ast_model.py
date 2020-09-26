import argparse
import sys

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Sequential

from asts.ast_vectorizer import ASTVectorizer

sys.path.append('..')
from tools.datatools import load_asts, load_all_labels
from tools.generators import Generator


def main(args):
    print('Settings:')
    print(str(args)[10:-1])
    train_x, dev_x, test_x = load_asts(args.data) # Load asts and labels
    train_y, dev_y, test_y = load_all_labels(args.data)
    if not args.no_recalc: # Recalculate the vectors for each ast
        vectorizer = ASTVectorizer(args.ngram, args.features, head_only=args.head)
        print('Training vectorizer')
        vectorizer.fit(train_x) # Train vectoriser on training set
        print('Fitting')
        train_x = vectorizer.transform(train_x) # Convert train, dev and test and save them to memmaps.
        fp = np.memmap('vectors/train.mm', dtype='float32', mode='w+', shape=(50000, args.features))
        fp[:] = train_x[:]
        del fp # Flushes memmap to file
        dev_x = vectorizer.transform(dev_x)
        fp = np.memmap('vectors/dev.mm', dtype='float32', mode='w+', shape=(25000, args.features))
        fp[:] = dev_x[:]
        del fp

        test_x = vectorizer.transform(test_x)
        fp = np.memmap('vectors/test.mm', dtype='float32', mode='w+', shape=(25000, args.features))
        fp[:] = test_x[:]
        del fp

        if args.mode == 'c': # Use minmax scaler for raw counts
            rescale(args.features)

    train = np.memmap('vectors/train.mm', dtype='float32', mode='r', shape=(50000, args.features))
    dev = np.memmap('vectors/dev.mm', dtype='float32', mode='r', shape=(25000, args.features))
    test = np.memmap('vectors/test.mm', dtype='float32', mode='r', shape=(25000, args.features))
    features = args.features

    train = Generator(train, train_y, args.batch) # Use generator to train from memmap
    dev = Generator(dev, dev_y, args.batch)
    callback_list = [EarlyStopping(monitor='val_acc', patience=10),
                     ModelCheckpoint(filepath='ast_model.h5', monitor='val_acc', save_best_only=True),
                     ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=3)]
    model = Sequential()
    model.add(Dense(1000, activation='relu', input_shape=(features,)))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])

    if not args.skip:
        model.fit(train, epochs=1000, validation_data=dev, shuffle=True, callbacks=callback_list)

    model.load_weights('ast_model.h5')
    model.evaluate(dev)
    predict_veca = np.memmap('vectors/dev_ast.mm', dtype='float32', mode='w+', shape=(25000, 1000))
    predict_veca[:] = model.predict(dev)[:] # Save dev and test predictions to file for ensemble.

    predict_vecb = np.memmap('vectors/test_ast.mm', dtype='float32', mode='w+', shape=(25000, 1000))
    predict_vecb[:] = model.predict(test)[:]
    del predict_veca, predict_vecb


def rescale(num_features): # Rescales feature counts between 0 and 1
    scaler = MinMaxScaler()

    train = np.memmap('vectors/train.mm', dtype='float32', mode='r+', shape=(50000, num_features))
    dev = np.memmap('vectors/dev.mm', dtype='float32', mode='r+', shape=(25000, num_features))
    test = np.memmap('vectors/test.mm', dtype='float32', mode='r+', shape=(25000, num_features))
    train[:] = scaler.fit_transform(train)
    dev[:] = scaler.transform(dev)
    test[:]=scaler.transform(test)
    del train, dev, test


def remove_invariants(num_features):  # Removes features with zero variance
    remover = VarianceThreshold()  # Only remove 0 variance by default
    train = np.memmap('vectors/train.mm', dtype='float32', mode='r', shape=(50000, num_features))
    dev = np.memmap('vectors/dev.mm', dtype='float32', mode='r', shape=(25000, num_features))
    train = remover.fit_transform(train)
    dev = remover.transform(dev)
    return train, dev, len(train[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('-d',
                        '--data',
                        type=str,
                        default=r'../data_dir',
                        help='Path to the data directory.')

    parser.add_argument('-m',
                        '--mode',
                        type=str,
                        default='b',
                        help='Mode for the vectorizer: Options are b (binary), c (count), t (TF-IDF). Default = b',
                        )
    parser.add_argument('-n',
                        '--ngram',
                        type=int,
                        default=1,
                        help='Node n-gram size. Default = 1'
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
    parser.add_argument('-s',
                        '--skip',
                        help='Whether to skip training and just load model from file. Default = False',
                        action='store_true')
    parser.add_argument('-u',
                        '--results',
                        help='Whether to write dev and train predictions to file. Default = False',
                        action='store_true')
    parser.add_argument('-i',
                        '--invariant',
                        help='Whether to remove invariant features from vectors Default = False',
                        action='store_true')
    parser.add_argument('-e',
                        '--head',
                        help='Whether to only use node heads. Default = False',
                        action='store_true')
    args = parser.parse_args()
    main(args)
