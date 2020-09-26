import argparse
import sys

sys.path.append('..')
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from tools.datatools import load_all_labels, load_all_pids, write_predictions
from tools.scorer import Scorer
from sklearn.model_selection import StratifiedKFold

# Loads all the predictions from the 5 models, both dev and test.

ngram = np.array(np.memmap('../ngrams/vectors/dev_ngram.mm', dtype='float32', mode='r', shape=(25000, 1000)))
ast = np.array(np.memmap('../asts/vectors/dev_ast.mm', dtype='float32', mode='r', shape=(25000, 1000)))
style = np.array(
    np.memmap('../stylometry/vectors/dev_style.mm', dtype='float32', mode='r', shape=(25000, 1000)))
bert = np.array(np.memmap('../bert/vectors/dev_bert.mm', dtype='float32', mode='r', shape=(25000, 1000)))
word = np.array(np.memmap('../words/vectors/dev_word.mm', dtype='float32', mode='r', shape=(25000, 1000)))
bert = softmax(bert, axis=1)
scorer = Scorer(r"../data_dir/dev.csv")
_, dev_y, _ = load_all_labels('../data_dir')
_, dev_pid, test_pid = load_all_pids('../data_dir')
ngram_test = np.array(np.memmap('../ngrams/vectors/test_ngram.mm', dtype='float32', mode='r', shape=(25000, 1000)))
ast_test = np.array(np.memmap('../asts/vectors/test_ast.mm', dtype='float32', mode='r', shape=(25000, 1000)))
style_test = np.array(
    np.memmap('../stylometry/vectors/test_style.mm', dtype='float32', mode='r', shape=(25000, 1000)))
bert_test = np.array(np.memmap('../bert/vectors/test_bert.mm', dtype='float32', mode='r', shape=(25000, 1000)))

word_test = np.array(
    np.memmap('../words/vectors/test_word.mm', dtype='float32', mode='r', shape=(25000, 1000)))
bert_test = softmax(bert_test, axis=1)


def main(args):
    print('Settings:')
    print(str(args)[10:-1])
    print()
    print(run_prediction(ngram, 'n-gram'))
    print(run_prediction(bert, 'CodeBERT'))
    print(run_prediction(word, 'Word'))
    print(run_prediction(ast, 'AST'))
    print(run_prediction(style, 'Stylometry'))
    print()
    models_to_use = []
    test_models_to_use = []
    ensemble_weights = []
    if not args.ngram and not args.bert and not args.ast and not args.word and not args.style:
        print("No models were specified for ensembling.")
        exit()
    # concatenated = np.empty(shape=(25000, 5000))
    models = [ngram, bert, word, ast, style]
    test_models = [ngram_test, bert_test, word_test, ast_test, style_test]
    to_use = [args.ngram, args.bert, args.word, args.ast, args.style]
    for i in range(len(to_use)):  # Builds list of required models in ensemble.
        if to_use[i]:
            models_to_use.append(models[i])
            test_models_to_use.append(test_models[i])

    concatenated = np.empty(shape=(25000, 1000 * len(models_to_use)))
    for i in range(len(models_to_use)):  # Concatenates used models into larger vector for Logistic Regression Ensemble.
        concatenated[:, i * 1000:(i + 1) * 1000] = models_to_use[i][:, :]
    if not args.min and not args.evo and not args.lr:
        print("No ensembling techniques were specified.")
    if args.min:  # Weighted average ensemble
        x = ensemble(models_to_use, mode='min')  # Run weighted average ensemble optimization using Powell.
        # , ast[:17500], word[:17500], style[:17500]
        # [ngram, bert, ast, word, style] 95.616 with low accuracy ast
        ensemble_weights = x[1]
        print(f"Weights: {x[1]}")
        print(f"Accuracy: {x[0]}")
        print()
    if args.evo:
        x = ensemble(models_to_use,
                     mode='evo')  # Run weighted average ensemble optimization using Differential Evolution.
        ensemble_weights = x[1]
        print(f"Weights: {x[1]}")
        print(f"Accuracy: {x[0]}")
        print()
    if args.lr:
        # Logistic regression stacking ensemble.
        print("Running logistic regression..")
        skf = StratifiedKFold(n_splits=5)  # Using stratified k-fold cross validation to extend validation set.
        accuracies = []
        for train_index, test_index in skf.split(concatenated, dev_y):  # Fit model 5 times on each fold set.
            model = LogisticRegression(max_iter=100, verbose=1, n_jobs=-1)
            model.fit(concatenated[train_index], dev_y[train_index])
            predictions = model.predict(concatenated[test_index])
            accuracies.append(accuracy_score(dev_y[test_index], predictions))
        acc = 0
        print()
        for a in accuracies:
            print(f"Accuracy: {a}")  # Print accuracies of each fold.
            acc += a
        print(f"Average Accuracy: {acc / 5}")  # Average accuracy.
        print()
        print("Training final predictor")
        model = LogisticRegression(max_iter=100, verbose=1,
                                   n_jobs=-1)  # Train final predictor on entire validation set.
        model.fit(concatenated, dev_y)
        concatenated_test = np.empty(shape=(25000, 1000 * len(models_to_use)))  # Concatenated test dataset
        for i in range(len(test_models_to_use)):
            concatenated_test[:, i * 1000:(i + 1) * 1000] = test_models_to_use[i][:, :]
        test_predictions = model.predict(concatenated_test)  # Run predictions from final predictor on test set.
        write_predictions(test_predictions, test_pid, 'test_stackensemble_result')  # Write results to file.
        exit()
    #############################
    # Write n-gram model only results to file.
    write_predictions(ngram.argmax(-1), dev_pid, 'dev_ngram_result')
    write_predictions(ngram_test.argmax(-1), test_pid, 'test_ngram_result')

    #############################
    # Write weighted average ensemble results to file.
    if args.min or args.evo:
        ensembled_models = np.zeros((25000, 1000))
        ensembled_test_models = np.zeros((25000, 1000))
        for i in range(len(models_to_use)):
            ensembled_models += models_to_use[i] * ensemble_weights[i]
            ensembled_test_models += test_models_to_use[i] * ensemble_weights[i]
        write_predictions(ensembled_models.argmax(-1), dev_pid, 'dev_ensemble_result')
        write_predictions(ensembled_test_models.argmax(-1), test_pid, 'test_ensemble_result')


def run_prediction(data, name):  # Get accuracy of individual dev file.
    result = data.argmax(-1)
    return (f'{name} accuracy: {scorer.accuracy(result)}')


def ensemble(models, mode='min'):  # Runs the weighted average optimisation techniques.
    models = np.array(models)
    if mode == 'min':
        print("Running Powell minimization.")
        x = minimize(objective, np.ones(len(models)), method='powell', args=models).x
    elif mode == 'evo':
        print("Running differential evolution.")
        bounds = [(0.0, 1.0)] * len(models)
        x = differential_evolution(objective, bounds=bounds, args=[models], maxiter=1000, tol=1e-7, workers=-1).x
    else:
        raise Exception(f'type must be \'min\' or \'evo\' but was {mode}.')
    x /= np.sum(x)
    if len(models) == 1:
        x = [x]
    return (1 - objective(x, models)), x


def objective(x, predictions):  # Weighted Average objective for highest accuracy / lowest inaccuracy.
    ensemble = np.zeros((25000, 1000))
    for i in range(len(x)):
        ensemble += predictions[i] * x[i]
    return 1.0 - accuracy_score(dev_y, ensemble.argmax(-1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    parser.add_argument('-ngram',
                        help='Whether to use n-gram model results. Default = False',
                        action='store_true')
    parser.add_argument('-bert',
                        help='Whether to use CodeBERT model results. Default = False',
                        action='store_true')
    parser.add_argument('-ast',
                        help='Whether to use AST model results. Default = False',
                        action='store_true')
    parser.add_argument('-word',
                        help='Whether to use Bag of Words model results. Default = False',
                        action='store_true')
    parser.add_argument('-style',
                        help='Whether to use Stylometry model results. Default = False',
                        action='store_true')
    parser.add_argument('-min',
                        help='Whether to use Powell optimisation Weighted Average ensemble. DO NOT USE WITH EVO ARGUMENT! Default = False',
                        action='store_true')
    parser.add_argument('-evo',
                        help='Whether to use differential Evolution Weighted Average ensemble. DO NOT USE WITH MIN ARGUMENT! Default = False',
                        action='store_true')
    parser.add_argument('-lr',
                        help='Whether to use Logistic Regression ensemble. Default = False',
                        action='store_true')

    args = parser.parse_args()
    main(args)
