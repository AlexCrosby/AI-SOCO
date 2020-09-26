import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def decode_one_hot(one_hot):
    return one_hot.index(1)


def encode_one_hot(uid):
    # 1000 dimension vector because there are 1000 uids
    one_hot = np.zeros(1000, dtype='bool')
    one_hot[uid] = 1
    return one_hot


def load_labels(file):  # Loads all the labels and stores them in a dictionary based on their pid as key
    # file = open(os.path.join(r'data_dir', '{}.csv'.format(file)))
    file = open(file)

    labels = list(csv.reader(file))[1:]

    labels = [[int(x), int(y)] for x, y in labels]
    labels.sort(key=lambda x: x[1])  # sort by pid in second column
    file.close()
    return zip(*labels)  # uid,pid sorted by pid


def load_files(dir, bytes, test=False):
    data = []
    if not test:
        file_list = [int(x) for x in os.listdir(dir)]
        file_list.sort()
    else:
        file_list = load_test_pids(os.path.join(dir, '../..'))
    for file_name in file_list:
        if not bytes:
            with open(os.path.join(dir, str(file_name)), encoding='utf-8')as f:
                data.append(f.read())

        else:
            with open(os.path.join(dir, str(file_name)), 'rb')as f:
                data.append(f.read())

    return data


# def prep_inputs(data_dir, bytes=False):
#     warnings.warn("prep_inputs is deprecated, please use load_data instead.")
#     train_data = load_datax(type, bytes)
#     train_label, train_pid = load_labels(r'../data_dir/{}.csv'.format(type))
#     return list(train_pid), train_data, np.array(
#         list(train_label))  # three parallel lists of data and labels and pids for training


def load_all_labels(data_dir):
    train_labels, _ = load_labels(os.path.join(data_dir, 'train.csv'))
    dev_labels, _ = load_labels(os.path.join(data_dir, 'dev.csv'))
    # test_labels, _ = load_labels(os.path.join(data_dir, 'test.csv'))
    test_labels = []

    return np.array(train_labels), np.array(dev_labels), np.array(test_labels)


def load_all_pids(data_dir):
    _, train_pids = load_labels(os.path.join(data_dir, 'train.csv'))
    _, dev_pids = load_labels(os.path.join(data_dir, 'dev.csv'))
    test_pids = load_test_pids(data_dir)
    return list(train_pids), list(dev_pids), test_pids


def load_data(data_dir, bytes=False, preprocess=False):
    # raw = orginal data
    # pre = removed comments and definitions

    train_label, _ = load_labels(os.path.join(data_dir, '{}.csv'.format('train')))
    dev_label, dev_pid = load_labels(os.path.join(data_dir, '{}.csv'.format('dev')))
    if preprocess:
        data_dir = os.path.join(data_dir, 'pre')
    else:
        data_dir = os.path.join(data_dir, 'raw')
    train_data = load_files(os.path.join(data_dir, 'train'), bytes)
    dev_data = load_files(os.path.join(data_dir, 'dev'), bytes)
    test_data = load_files(os.path.join(data_dir, 'test'), bytes, test=True)
    test_label = None
    test_pid = load_test_pids(os.path.join(data_dir, '..'))
    return train_data, np.array(train_label), dev_data, np.array(dev_label), np.array(
        dev_pid), test_data, test_label, test_pid


def write_predictions(pred, dev_ids, name):
    fp = ('{}.csv'.format(name))
    print('Saving prediction to {}'.format(fp))
    if sys.version_info >= (3, 0, 0):
        # This resolves windows vs linux csv issues where windows adds an extra new line between each csv row.
        fp = open(fp, 'w', newline='')
    else:
        fp = open(fp, 'wb')

    with fp:
        writer = csv.writer(fp)
        writer.writerow(['uid', 'pid'])

        for p, i in zip(pred, dev_ids):
            writer.writerow([p, i])
        fp.close()


def plot_history(history):
    # This plotter is adapted from Deep Learning with Python by Francois Chollet
    # print(history.history.keys())
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, color='#00c3e3', label='Training Accuracy')
    plt.plot(epochs, val_acc, color='#ff4554', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, color='#00c3e3', label='Training Loss')
    plt.plot(epochs, val_loss, color='#ff4554', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def load_asts(data_dir):
    ast_file = os.path.join(data_dir, 'ast', 'ast.json')
    with open(ast_file, 'r')as f:
        data = json.load(f)
        tr = data['train']
        de = data['dev']
        te = data['test']
        tr = reorder_ast(tr)
        de = reorder_ast(de)
        te = reorder_ast(te)
        return tr, de, te


def reorder_ast(dict): # Turns each dict into the correctly ordered list.
    ordered_keys = (sorted(dict, key=lambda k: int(k)))
    return [dict[x] for x in ordered_keys]


def load_test_pids(data_dir):
    file = os.path.join(data_dir, 'unlabeled_test.csv')
    file = open(file)
    pids = list(csv.reader(file))[1:]
    file.close()
    pids = [int(x[1]) for x in pids]
    return pids
