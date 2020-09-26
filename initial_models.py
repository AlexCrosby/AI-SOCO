import sys

sys.path.append('..')
from tools.datatools import load_data
from joblib import dump, load
from vectorizer import Vectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import time

from sklearn.preprocessing import MinMaxScaler

# Parameters must be manually edited within this file.
# This was the first model(s) created.


def main():
    start_time = time.time()
    ################# ML MODEL#########################
    # model = MultinomialNB()
    # model = LogisticRegression(max_iter=100, verbose=1, n_jobs=-1)
    # model = KNeighborsClassifier(n_neighbors=25, n_jobs=-1)
    model = SVC()
    ################ VECTOR MODEL ####################
    all_models = ['alpha_vec', 'char_vec', 'tfidfsk_vec']
    vector_model = all_models[0]
    vectorizer = Vectorizer(vector_model[:-4])
    #################################################
    file = str(type(model))[:-2]
    file = vector_model + '_' + file.split('.')[-1]
    print(file)


    recalc_vectors = True
    vectorize_only = False

    if vectorize_only:
        recalc_vectors = True
    #################### load data#######################
    section_time = time.time()

    train_x, train_y, dev_x, dev_y, _, _, _, _ = load_data('data_dir')

    print("--- %s seconds to load data ---" % (time.time() - section_time))
    print("--- %s seconds elapsed ---" % (time.time() - start_time))
    ####################recalc vectors #########################
    section_time = time.time()
    if recalc_vectors:
        print("Recomputing vectors...")
        train_x = vectorizer.vectorize(train_x)
        dev_x = vectorizer.vectorize(dev_x)
        dump(train_x, 'vectorised_data/{}_train.joblib'.format(vector_model), compress=True)
        dump(dev_x, 'vectorised_data/{}_dev.joblib'.format(vector_model), compress=True)
        print("--- %s seconds to recalculate vectors ---" % (time.time() - section_time))

        ############ load vectors########################
    else:
        print("Loading precomputed vectors...")

        train_x = load('vectorised_data/{}_train.joblib'.format(vector_model))

        dev_x = load('vectorised_data/{}_dev.joblib'.format(vector_model))
        print("--- %s seconds to load precalculated vectors ---" % (time.time() - section_time))
    print("--- %s seconds elapsed ---" % (time.time() - start_time))
    print("Data preparation finished.")

    ##################### vectors ready ###############################

    if vectorize_only:
        exit("Vectorised only.")
        print("--- %s seconds elapsed ---" % (time.time() - start_time))
        print("Program finished.")

    #     ############################# train model #####################
    section_time = time.time()

    print("Training model...")
    model.fit(train_x, train_y)
    dump(model, 'models/{}.joblib'.format(file), compress=True)
    print("--- %s seconds to train model ---" % (time.time() - section_time))


    # ########################     model ready start fitting #################################
    section_time = time.time()

    print("Predicting data...")
    predictions = model.predict(dev_x)
    print('Accuracy: {}'.format(accuracy_score(dev_y, predictions)))
    print("--- %s seconds to predict data ---" % (time.time() - section_time))
    print("--- %s seconds elapsed ---" % (time.time() - start_time))
    print("Program finished.")


if __name__ == '__main__':
    main()
