import sys
from string import printable

sys.path.append('..')
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer

from stylometry.stylometry_vectorizer import Stylometry

# Contains multiple smaller vectorizer types for code
class Vectorizer:

    def __init__(self, type):
        self.vector_dict = {
            'alpha': self.alpha_vec,
            'char': self.char_vec,
            'tfidfsk': self.tfidfsk_vec,
            'tfidfskc': self.tfidfskc_vec,
            'tfidfskt': self.tfidfskt_vec,
            'token': self.tokenizer,
            'lexical': self.lexical

        }
        self.vectorizer = self.vector_dict[type.lower()]
        self.tfidf = None
        self.token = None

    def vectorize(self, data, data2=None):
        if data2 == None:
            return self.vectorizer(data)
        else:
            return self.lexical(data, data2)

    @staticmethod
    def alpha_vec(data):  # Only counts usage of each a-z character. All symbols ignored
        output = []
        alpha_dict = {}
        for i in range(26):
            alpha_dict[chr(i + 97)] = i
        for code in data:
            code = code.lower()
            vector = np.zeros(26)
            for char in code:
                if char in alpha_dict:
                    vector[alpha_dict[char]] += 1
            output.append(vector)
        return output

    @staticmethod
    def char_vec(data):  # Taken from baseline is a count of every printable character used.
        printable_dict = {char: idx for idx, char in enumerate(printable)}
        output = []
        for code in data:

            vector = [0] * len(printable_dict)
            for char in code:
                if char in printable_dict:
                    vector[printable_dict[char]] += 1
            output.append(vector)
        return output

    def tfidfsk_vec(self, data, analyzer='word', tokenizer=None):
        if len(data) == 50000:  # Only the training set has 50k entries, dev and test have 25k
            self.tfidf = TfidfVectorizer(max_features=10000, sublinear_tf=True, analyzer=analyzer)
            return self.tfidf.fit_transform(data).astype('float32').toarray()
        else:
            if self.tfidf is None:
                raise Exception("TF-IDF model has not been fitted.")
            return self.tfidf.transform(data).astype('float32').toarray()

    def tfidfskc_vec(self, data):
        return self.tfidfsk_vec(data, analyzer='char')

    def tfidfskt_vec(self, data):
        return self.tfidfsk_vec(data, tokenizer=self.tokenizer)

    def tokenizer(self, data):
        if len(data) == 50000:
            self.token = Tokenizer(num_words=10000, char_level=True)
            self.token.fit_on_texts(data)
            n = self.token.texts_to_sequences(data)
            print(len(n[0]))
            return n
        else:
            print("working")
            if self.token is None:
                raise Exception("Tokenizer model has not been fitted.")
            n = self.token.texts_to_sequences(data)
            print("done")
            return n

    def lexical(self, data, preprocessed_data):  # Stylometry and character vectorizer.
        s = Stylometry()
        vectors = [[] for _ in range(len(data))]
        for i in range(len(data)):
            vectors[i] = (s.parse(data[i], preprocessed_data[i]))
            vectors[i] += self.char_vec([data[i]])[0]
        return np.array(vectors).astype('float32')
