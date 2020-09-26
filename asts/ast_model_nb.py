import argparse
import sys

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

from asts.ast_vectorizer import ASTVectorizer

sys.path.append('..')
from tools.datatools import load_asts, load_all_labels


def main(args):
    print('Settings:')
    print(str(args)[10:-1])
    train_x, dev_x, test_x = load_asts(args.data)
    train_y, dev_y, test_y = load_all_labels(args.data)
    if not args.no_recalc:
        vectorizer = ASTVectorizer(args.ngram, args.features, head_only=args.head)
        vectorizer.fit(train_x)
        print('Training vectorizer')
        train_x = vectorizer.transform(train_x)
        print('Fitting')
        fp = np.memmap('vectors/train.mm', dtype='float32', mode='w+', shape=(50000, args.features))
        fp[:] = train_x[:]
        del fp
        dev_x = vectorizer.transform(dev_x)
        fp = np.memmap('vectors/dev.mm', dtype='float32', mode='w+', shape=(25000, args.features))
        fp[:] = dev_x[:]
        del fp
        if args.mode == 'c':
            rescale(args.features)

    train = np.array(np.memmap('vectors/train.mm', dtype='float32', mode='r', shape=(50000, args.features)))

    print("Training model...")
    model = MultinomialNB()
    model.fit(train, train_y)
    del train
    dev = np.array(np.memmap('vectors/dev.mm', dtype='float32', mode='r', shape=(25000, args.features)))
    print("Predicting data...")
    predictions = model.predict(dev)
    print('Accuracy: {}'.format(accuracy_score(dev_y, predictions)))


def rescale(num_features):
    scaler = MinMaxScaler()
    train = np.memmap('vectors/train.mm', dtype='float32', mode='r+', shape=(50000, num_features))
    dev = np.memmap('vectors/dev.mm', dtype='float32', mode='r+', shape=(25000, num_features))
    train[:] = scaler.fit_transform(train)
    dev[:] = scaler.transform(dev)
    del train, dev


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
