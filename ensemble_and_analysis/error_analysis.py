import sys

import numpy as np
from scipy.special import softmax

sys.path.append('..')
from tools.scorer import Scorer

ngram = np.array(
    np.memmap('../ngrams/vectors/dev_ngram.mm', dtype='float32', mode='r', shape=(25000, 1000))).argmax(-1)
ast = np.array(np.memmap('../asts/vectors/dev_ast.mm', dtype='float32', mode='r', shape=(25000, 1000))).argmax(-1)
style = np.array(
    np.memmap('../stylometry/vectors/dev_style.mm', dtype='float32', mode='r', shape=(25000, 1000))).argmax(-1)
bert = np.array(np.memmap('../bert/vectors/dev_bert.mm', dtype='float32', mode='r', shape=(25000, 1000)))
word = np.array(
    np.memmap('../words/vectors/dev_word.mm', dtype='float32', mode='r', shape=(25000, 1000))).argmax(-1)
bert = softmax(bert, axis=1).argmax(-1)


# Investigates what models are capable of making which predictions. Provides some basic statistics and list of files
# to be analysed.
# scorer in tools directory provides the enumeration.
def main():
    # Select a user for incorrect results analysis and correct results analysis.
    incorrect_user = 672
    correct_user = 986

    scorer = Scorer(r"../data_dir/dev.csv")
    print('Individual model results:')
    print(f'Ngram accuracy: {scorer.accuracy(ngram)}')
    print(f'AST accuracy: {scorer.accuracy(ast)}')
    print(f'Stylometry accuracy: {scorer.accuracy(style)}')
    print(f'CodeBERT accuracy: {scorer.accuracy(bert)}')
    print(f'Word accuracy: {scorer.accuracy(word)}')
    print()

    ngram_err = scorer.errors(ngram)
    ast_err = scorer.errors(ast)
    style_err = scorer.errors(style)
    bert_err = scorer.errors(bert)
    word_err = scorer.errors(word)
    print("Intersection of all incorrect guesses:")
    intersect = ngram_err[1].intersection(ast_err[1], style_err[1], bert_err[1], word_err[1])
    print(len(intersect))
    consistent_incorrect = intersect
    scorer.analyse_incorrect(consistent_incorrect, incorrect_user)
    print()
    print("What ngrams correctly guess that no other model can:")
    ngrams_only = ngram_err[0].difference(ast_err[0], style_err[0], bert_err[0], word_err[0])
    print(len(ngrams_only))
    scorer.analyse_correct(ngrams_only, correct_user)


if __name__ == '__main__':
    main()
