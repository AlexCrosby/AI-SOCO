import sys

sys.path.append('..')
import argparse
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tools.datatools import load_data
from tools.generators import Generator, GeneratorX

 # Bag-of-Words model

def main(args):
    train_x, train_y, dev_x, dev_y, _, test_x, _, _ = load_data('../data_dir')
    features = 60770  # Number of unique words in training set
    if args.no_recalc:
        vectorizer = CountVectorizer(binary=True)
        # vectorizer = TfidfVectorizer()

        # Convert train, dev and test to 60770-D vectors
        train_x = vectorizer.fit_transform(train_x).astype('float32').toarray()  # Convert train set to vector
        features = train_x.shape[1]
        t = np.memmap('vectors/train.mm', dtype='float32', mode='w+', shape=(50000, features))
        t[:] = train_x[:]
        del t, train_x
        dev_x = vectorizer.transform(dev_x).astype('float32').toarray()
        d = np.memmap('vectors/dev.mm', dtype='float32', mode='w+', shape=(25000, features))
        d[:] = dev_x[:]
        del d, dev_x
        test_x = vectorizer.transform(test_x).astype('float32').toarray()
        te = np.memmap('vectors/test.mm', dtype='float32', mode='w+', shape=(25000, features))
        te[:] = test_x[:]
        del te, test_x
    t = np.memmap('vectors/train.mm', dtype='float32', mode='r', shape=(50000, features))
    d = np.memmap('vectors/dev.mm', dtype='float32', mode='r', shape=(25000, features))
    te = np.memmap('vectors/test.mm', dtype='float32', mode='r', shape=(25000, features))
    # Setup generators
    train = Generator(t, train_y, 128)
    dev = Generator(d, dev_y, 128)
    test = GeneratorX(te, 128)
    #  Model
    callback_list = [EarlyStopping(monitor='val_acc', patience=5),
                     ModelCheckpoint(filepath='word_model.h5', monitor='val_acc', save_best_only=True),
                     ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=3)]

    opt = RMSprop(learning_rate=0.001)
    model = Sequential()
    model.add(Dense(500, activation='relu', input_shape=(features,)))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['acc'])
    model.summary()
    model.fit(train, epochs=1000, validation_data=dev, callbacks=callback_list)
    model.load_weights('word_model.h5')
    model.evaluate(dev)
    # Write dev and test predictions to file.
    predict_vec = np.memmap('vectors/dev_word.mm', dtype='float32', mode='w+', shape=(25000, 1000))
    predict_vec[:] = model.predict(dev)[:]
    del predict_vec
    predict_vec2 = np.memmap('vectors/test_word.mm', dtype='float32', mode='w+', shape=(25000, 1000))
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
