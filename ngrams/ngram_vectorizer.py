import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class NgramVectorizer:

    def __init__(self, size=1, matrix=True, count_mode='b', max_features=-1):
        # matrix: whether a vector(matrix) or list representation
        # count_mode b ,c or t
        # b = binary words
        # c = count words
        # t= tfidf words
        self.ngram_size = size
        self.matrix = matrix
        self.count_mode = count_mode
        self.position_dict = {}
        self.count_dict = {}
        self.position_counter = 0
        self.max_features = max_features
        self.tfidf = None

    def fit(self, code_set):
        num_ngrams = 0
        if self.count_mode is not 't':  # For raw and binary count.
            for code in code_set:  # For each source code.
                for i in range(len(code)):  # Iterate though character by character.
                    if i + self.ngram_size > len(code):  # End when you can't get any more full-size n-grams.
                        break
                    ngram = code[i:i + self.ngram_size]  # Get n-gram of size ngram_size.
                    self.count_dict[ngram] = self.count_dict.get(ngram, 0) + 1  # Add it to the count dict.
            count_list = sorted(self.count_dict, key=self.count_dict.get, reverse=True)  # Sort all ngrams by frequency.
            if self.max_features > 0:  # If below zero, assume infinite max_features is infinite.
                count_list = count_list[:self.max_features]  # Truncate to max_features
            self.count_dict = {key: self.count_dict[key] for key in
                               count_list}  # Remove truncated features from count_dict for memory saving.
            for ngram in count_list:  # Set up a position index dictionary to speed up lookups.
                self.position_dict[ngram] = self.position_counter
                self.position_counter += 1
            num_ngrams = self.position_counter
        elif self.count_mode is 't':  # TF-IDF mode.
            if self.max_features > 0:  # Set up and run sklearn TfidfVectorizer using the tfidf_tokenizer method.
                self.tfidf = TfidfVectorizer(tokenizer=self.tfidf_tokenizer, max_features=self.max_features)
            else:
                self.tfidf = TfidfVectorizer(tokenizer=self.tfidf_tokenizer)
            self.tfidf.fit(code_set)
            num_ngrams = self.tfidf.vocabulary_.__len__()
        return num_ngrams  # Count of n-grams found (should be same as max features).

    def transform(self, code_set):  # picks the transform method based on class settings.
        if not self.matrix:
            return self.transform_list(code_set)
        if self.matrix:
            if self.count_mode is not 't':
                return self.transform_matrix(code_set)
            elif self.count_mode is 't':
                return self.tfidf.transform(code_set).toarray()

    def transform_list(self, code_set):  # Returns all ngrams as a list for debugging purposes.
        all_outputs = []
        for code in code_set:
            output = []
            for i in range(len(code)):
                if i + self.ngram_size > len(code):
                    break
                ngram = code[i:i + self.ngram_size]
                ngram_number = self.position_dict.get(ngram)
                if ngram_number is None:
                    continue
                output.append(ngram_number)
            all_outputs.append(output)
        return all_outputs

    def transform_matrix(self, code_set):  # For binary and count vectorization
        all_outputs = []
        for code in code_set:
            output = np.zeros(self.position_counter, dtype='float32')
            for i in range(len(code)):
                if i + self.ngram_size > len(code):  # Get n-grams as in fit.
                    break
                ngram = code[i:i + self.ngram_size]
                position = self.position_dict.get(ngram, None)
                if position is None:
                    continue
                else:
                    if self.count_mode == 'b':  # Binary caps at 1.
                        output[position] = 1
                    elif self.count_mode == 'c':
                        output[position] += 1
            all_outputs.append(output)
        return np.asarray(all_outputs, dtype='float32')

    def ngram_count(self):
        return self.position_counter

    def tfidf_tokenizer(self, code):  # tokenization method used for sklearn TfidfVectorizer.
        output = []
        for i in range(len(code)):
            if i + self.ngram_size > len(code):
                break
            ngram = code[i:i + self.ngram_size]
            output.append(ngram)
        return output
