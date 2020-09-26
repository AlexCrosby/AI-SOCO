import argparse
import csv
import sys
from collections import Counter

sys.path.append('..')
from sklearn.metrics import accuracy_score


# This file is an expansion on the scorer.py script provided by the AI-SOCO 2020 team hence uses a lot of common
# methods.
# This can be run directly, passing files as arguments or as an import. It also contains some methods for error
# analysis.
class Scorer:

    def __init__(self, gold_file):
        self.gold = self.load_labels(gold_file)
        self.gold = sorted(self.gold, key=lambda elem: elem[1])  # Sorts list by pid
        self.file = [elem[1] for elem in self.gold]  # List of pids in numerical order
        self.gold = [elem[0] for elem in self.gold]  # Convert to list of just uids ordered by pid

    def load_labels(self, file_path):  # Loads files into a 2d list: uid | pid
        with open(file_path, 'r') as fp:
            reader = csv.reader(fp)
            labels = list(reader)
        assert (labels[0][0] == 'uid')
        assert (labels[0][1] == 'pid')
        labels = labels[1:]
        for i in range(len(labels)):
            labels[i][0] = int(labels[i][0])
            labels[i][1] = int(labels[i][1])
        return labels

    def accuracy(self, prediction):  # Predictions should be ordered by pid
        correct = 0
        for i in range(len(prediction)):
            if self.gold[i] == prediction[i]:
                correct += 1
        return (correct / len(prediction))

    def error_analysis(self, prediction):  # Simple error analysis and accuracy

        correct_pids, incorrect_pids = self.errors(prediction)

        print(f'Correct guesses: {len(correct_pids)}')
        print(f'Incorrect guesses: {len(incorrect_pids)}')
        print(f'Accuracy: {len(correct_pids) / len(prediction)}')
        self.analyse_incorrect(incorrect_pids)
        return correct_pids, incorrect_pids

    def analyse_incorrect(self, incorrect_pids, uid=-1):  # Analyses pids which were incorrectly guessed and which
        # authors were responsible for misclassified codes.
        answer_dict = self.load_dict()
        incorrect_user_guesses = [answer_dict[pid] for pid in incorrect_pids]

        incorrect_user_guesses_counter = Counter(incorrect_user_guesses)
        incorrect_user_guesses = sorted(incorrect_user_guesses_counter, key=incorrect_user_guesses_counter.get,
                                        reverse=True)

        print(f"Number of unique users who have at least one incorrectly predicted file: {len(incorrect_user_guesses)}")
        print(
            f"Number of unique users who have at least two incorrectly predicted files: {len([key for key, val in incorrect_user_guesses_counter.items() if val != 1])}")
        print(
            f"First most errors attributed to user {incorrect_user_guesses[0]}: {incorrect_user_guesses_counter[incorrect_user_guesses[0]]}")
        print(
            f"Second most errors attributed to user {incorrect_user_guesses[1]}: {incorrect_user_guesses_counter[incorrect_user_guesses[1]]}")
        print(
            f"Third most errors attributed to user {incorrect_user_guesses[2]}: {incorrect_user_guesses_counter[incorrect_user_guesses[2]]}")
        print("UIDs and incorrect counts:")
        print(incorrect_user_guesses_counter)

        if uid != -1:
            print(f"Incorrectly guessed files for user {uid}")
            print(sorted([pid for pid in incorrect_pids if answer_dict[pid] == uid]))

    def analyse_correct(self, correct_pids, uid=-1):  # Analyses pids which were correctly guessed and which
        # authors were responsible for correctly classified codes.
        answer_dict = self.load_dict()
        user_counter = [answer_dict[pid] for pid in correct_pids]
        correct_user_guesses_counter = Counter(user_counter)

        print('UIDs and correct counts:')
        print(correct_user_guesses_counter)

        if uid != -1:
            print(f"Correctly guessed files for user {uid}")
            print(sorted([pid for pid in correct_pids if answer_dict[pid] == uid]))

    def errors(self,
               prediction):  # creates two lists, one with correctly guessed pids, the other with incorrect guesses
        # from the overall predictions.
        correct_pids = []
        incorrect_pids = []
        for i in range(len(prediction)):
            if self.gold[i] == prediction[i]:
                correct_pids.append(self.file[i])
            else:
                incorrect_pids.append(self.file[i])
        return set(correct_pids), set(incorrect_pids)

    def load_dict(self):  # loads gold file as a dict to look up uid for each pid
        dict = {}
        with open(r"../data_dir/dev.csv", 'r') as fp:
            reader = csv.reader(fp)
            labels = list(reader)
            labels = labels[1:]
            for pair in labels:
                dict[int(pair[1])] = int(pair[0])
        return dict


if __name__ == '__main__':  # Taken from AI-SOCO scorer.
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_file')
    parser.add_argument('--pred_file')
    args = parser.parse_args()
    scorer = Scorer(args.gold_file)
    gold = scorer.load_labels(args.gold_file)
    pred = scorer.load_labels(args.pred_file)
    assert (len(gold) == len(pred))
    gold = sorted(gold, key=lambda elem: elem[1])
    pred = sorted(pred, key=lambda elem: elem[1])
    gold = [elem[0] for elem in gold]
    pred = [elem[0] for elem in pred]

    print('Accuracy: {}'.format(accuracy_score(gold, pred)))
