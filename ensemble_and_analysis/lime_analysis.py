import numpy as np
from joblib import load
from lime.lime_tabular import LimeTabularExplainer
from tensorflow.keras.models import load_model

from tools.datatools import load_all_pids

model = load_model('../ngrams/ngram_model.h5') # Loads trained n-gram neural net model.
_, dev_pids, _ = load_all_pids('../data_dir')

# This tried to explain n-gram results using LIME. This works in principle but is not finished. Requires a long time
# and large amounts of memory to run.

def main():
    pid = 9111 # Example file to run.
    file_index = dev_pids.index(pid) # Index of example file.

    dev = np.array(np.memmap('../ngrams/vectors/dev.mm', dtype='float32', mode='r', shape=(25000, 20000)), dtype='bool')

    vec = load('../ngrams/vectorizer.joblib') # Use the asts_old pre-trained n-gram vectorizer.
    names = sorted(vec.count_dict, key=vec.count_dict.get, reverse=True).copy() # List of ngrams to label each feature
    names = [str(name) for name in names]
    del vec
    file_vector= dev[file_index] # Gets the vector for the given pid
    class_names = list(range(1000))
    print("Training explainer...")
    # Explainer must be trained on the entire dev set to make inferences.
    explainer = LimeTabularExplainer(dev, feature_names=names, class_names=class_names, verbose=True,
                                     mode='classification', categorical_features=range(20000),
                                     categorical_names=names)  # Had to limit the training due to memory
    del dev
    print("Explaining...")
    np.random.seed(1) # generates explanation about given file in relation to specified labels.
    exp = explainer.explain_instance(file_vector, pred, num_features=25, labels=[672, 71], num_samples=10000)
    print(exp.as_list(label=672))
    a = exp.as_pyplot_figure(label=672)
    a.show()
    exp.save_to_file('analysis.html')


def pred(ary): # Use the n-gram model to make predictions.
    return (model.predict(ary))


if __name__ == '__main__':
    main()
