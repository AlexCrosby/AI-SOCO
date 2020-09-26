import os
from joblib import load
import sys

sys.path.append('..')
# Looks at the specific ngrams occuring in specific source codes.
def main():
    vec = load("../ngrams/vectorizer.joblib")
    names = set(sorted(vec.count_dict, key=vec.count_dict.get, reverse=True).copy())
    vector_ngrams = set(names)
    incorrect = [9111, 22447, 55924, 55957, 58962, 61223, 65929] # These files taken from error_analysis.py
    correct = [75518, 1488, 76085, 10690, 15601, 28443, 64741, 54892, 44971, 75861, 17867, 34071, 60042, 12406, 6170,
               91305, 10636, 19155] # These are from user 672 as discussed in report.


    for n in correct:
        vector_ngrams = vector_ngrams.intersection(get_ngrams(n)) # Common n-grams between all correctly guessed files.

    print(f'Common n-grams between correctly guessed files: {len(vector_ngrams)}') # Number of common ngrams between
    # files also used in the vector representations

    common_incorrect = vector_ngrams.intersection(get_ngrams(incorrect[0])) # ngrams in files 9111 also in all other
    # correctly guessed files

    print(f'Common n-grams between incorrect file and correct: {len(common_incorrect)}')

def load_file(n): # Reads a specific file to bytes.
    file = os.path.join('../data_dir/raw/dev', str(n))
    with open(file, 'rb')as f:
        return f.read()


def get_ngrams(n): # Get all n-grams from bytes.
    code = load_file(n)
    ngrams = set()
    for i in range(len(code)):
        if i + 6 > len(code):
            break
        ngrams.add(code[i:i + 6])
    return ngrams


if __name__ == '__main__':
    main()
