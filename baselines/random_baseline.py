import os
import csv
import random
import argparse
import sys

from sklearn.metrics import accuracy_score

# This is a baseline file provided by the AI-SOCO 2020 team and is not my own work. It may contain some slight
# differences made by myself however.
def load_labels(split):
    with open(os.path.join(args.data_dir, '{}.csv'.format(split)), 'r') as fp:
        reader = csv.reader(fp)
        problems = list(reader)
    problems = problems[1:]
    return problems


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data_dir')
    args = parser.parse_args()
    args.data_dir = r"..\data_dir"  # Manual file location added
    dev_csv = load_labels('dev')
    dev_labels, dev_ids = zip(*dev_csv)
    dev_labels = list(map(int, dev_labels))
    dev_ids = list(map(int, dev_ids))

    pred = [random.randint(0, 999) for _ in range(len(dev_labels))]

    print('Accuracy: {}'.format(accuracy_score(dev_labels, pred)))

    fp = '../baselines_predictions/random_baseline.csv'  # Added manual baseline file location

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
