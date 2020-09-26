from math import log

import numpy as np


class ASTVectorizer:

    def __init__(self, ngram_size, max_features, mode='b', head_only=False):
        self.ngram_size = ngram_size
        self.max_features = max_features
        self.mode = mode
        self.feature_count = {}  # A count of all features over all docs.
        self.feature_index = {}  # Index of each feature after trimming and sorting by total counts.
        self.feature_doc_count = {}  # Number of docs each ast node term appears in.
        self.doc_count = 0
        self.type_only = head_only

    def fit(self, train_data):
        self.doc_count = len(train_data)
        for train_ast in train_data:
            ngrams = self.get_ngrams(train_ast, self.ngram_size, self.type_only)  # Extracts ast nodes from each file
            for key in ngrams:  # Counts times ast node appears and number of documents node is in.
                self.feature_count[key] = self.feature_count.get(key, 0) + ngrams[key]
                self.feature_doc_count[key] = self.feature_doc_count.get(key, 0) + 1
        # All features collected, now must trim down to max_features sorted by number of keys
        trimmed_keys = sorted(self.feature_count, key=self.feature_count.get, reverse=True)[:self.max_features]
        all_keys = list(self.feature_count.keys())
        for key in all_keys:
            if key not in trimmed_keys:  # Remove unused keys from dictionaries to save memory.
                del self.feature_count[key]
                del self.feature_doc_count[key]
        # feature dict is now correct length with correct counts from train. Create index dict to speed up vector
        # building.

        for i in range(len(
                trimmed_keys)):  # Each n-gram corresponds to a specific index, this dict speeds the lookup process.
            self.feature_index[trimmed_keys[i]] = i
        # now training is done

    def transform(self, data):  # Method for transforming asts to vectors.
        if self.mode == 't':
            return self.tf_idf(data)  # TF-IDF MODE
        else:
            return self.transform_bc(data)  # Regular count + binary mode.

    @staticmethod
    def tf(count, total):  # Term frequency calculator.
        return count / total

    def idf(self, ngram):  # Inverse Document Frequency calculator/
        n = self.doc_count
        dft = self.feature_doc_count[ngram]
        return log(n / dft)

    @staticmethod
    def count_terms(ngrams):  # Counts the total number of node n-grams from node count dict.
        total = 0
        for n in ngrams:
            total += ngrams[n]
        return total

    def tf_idf(self, data):  # Perform TF-IDF calculations on all asts.
        output = []
        for ast in data:
            vector = np.zeros(self.max_features)  # Set up empty vector
            ngrams = self.get_ngrams(ast, self.ngram_size, head_only=self.type_only)  # Grab node n-grams.

            for n in ngrams:
                index = self.feature_index.get(n, None)  # Get index of specific a node n-gram in vector.
                if index is not None:  # Ignore features not assigned an index.
                    tf = self.tf(ngrams[n], self.count_terms(ngrams))
                    idf = self.idf(n)
                    vector[index] = tf * idf  # Calculate tf-idf for each node n-gram.
            output.append(vector)
        return np.array(output, dtype='float32')

    def transform_bc(self, data):  # Perform count vectorisation on all asts.
        output = []
        for ast in data:
            vector = np.zeros(self.max_features)  # Setup empty vector and get node n-grams.
            ngrams = self.get_ngrams(ast, self.ngram_size, self.type_only)
            for n in ngrams:
                index = self.feature_index.get(n, None)  # Get index and ignore unassigned.
                if index is not None:
                    if self.mode == 'c':
                        vector[index] = ngrams[n]  # Count node n-gram frequency.
                    elif self.mode == 'b':
                        vector[index] = 1  # Binary count maxes at 1.

            output.append(vector)
        return np.array(output, dtype='float32')

    @staticmethod
    def get_ngrams(ast_code, n=1, head_only=False):  # Extracts a dict of ngrams and their counts.

        level = 0  # Current depth in AST
        branch_dict = {}  # Saves the current branch we are traversing. This stored current level as the key and the
        # node as the value. This allows us to easily look up the parent/grandparents etc of a node at a given level
        # while traversing the current branch.
        item = ''  # Building the node string.
        ngrams = {}  # Count of each node n-gram
        for c in ast_code:  # Iterate through AST string character by character.
            if c is not '{' and c is not '}':
                item += c  # While we are traversing through the text of a node, save that node to item.
            else:  # We reach '{' or '}' which means we have reached the end of the current node.
                if len(item) > 0:  # Checks we actually have a recorded node.
                    branch_dict[level] = item
                    ngram = branch_dict[level]  # Get the current level and save the node to that level.
                    if n == 1:  # Unigram nodes
                        if head_only:  # Removes the code part of the node and only keeps the node type.
                            ngram = ASTVectorizer.remove_tail(ngram)  # Cuts off code segment of the node.
                        ngrams[ngram] = ngrams.get(ngram, 0) + 1  # Increase count for node n-gram.
                    elif n > 1:
                        if level >= n - 1:  # skips this level n-gram size requires higher levels.
                            # The reason is that the code searches upwards to retrieve parent nodes, there are no parent
                            # nodes for the root node for instance.
                            for i in (range(1, n)):  # Number of parent nodes to find.
                                ngram = branch_dict[
                                            level - i] + ':' + ngram  # Traverse up current branch and add parents to
                                # current node string if required.
                            if head_only:  # Again, cut off code section if required.
                                ngram = ASTVectorizer.remove_tail(ngram)
                            ngrams[ngram] = ngrams.get(ngram, 0) + 1  # And count.

                if c is '{':  # Go down a level and start new node
                    level += 1
                    item = ''
                elif c is '}':  # Go up a level and start new node.
                    level -= 1
                    item = ''
        return ngrams

    @staticmethod
    def remove_tail(ngram):
        nodes = ngram.split(':')  # Splits node n-gram into constituent nodes.
        head_nodes = [node.split()[1] for node in nodes]  # Removes code section.
        ngram = ':'.join(head_nodes)  # Rejoin nodes now with type only.
        return ngram
