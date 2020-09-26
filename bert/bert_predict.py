import sys

import numpy as np
from transformers import TFRobertaForSequenceClassification

sys.path.append('..')
from tools.generators import GeneratorBERT

# This scripts generates the probability scores for each piece of code using the fine-tuned bert model.

length = 200
model_path = "../data_dir/codebert/pytorch_model.bin"
dev_path = f"vectors/dev_{length}.mm"
train_path = f"vectors/train_{length}.mm"
test_path = f"vectors/test_{length}.mm"
mode = 3
batch_size = 250

empty = np.zeros(25000)
# Load data and setup generators.
dev = np.array(np.memmap(dev_path, dtype='int32', mode='r', shape=(25000, 3, length)))
test = np.array(np.memmap(test_path, dtype='int32', mode='r', shape=(25000, 3, length)))
train = np.array(np.memmap(train_path, dtype='int32', mode='r', shape=(50000, 3, length)))
dev = GeneratorBERT(dev, empty, batch_size, mode)
test = GeneratorBERT(test, empty, batch_size, mode)
train = GeneratorBERT(train, empty, batch_size, mode)
model = TFRobertaForSequenceClassification.from_pretrained(model_path, config='roberta-base', from_pt=True,
                                                           num_labels=1000)
# Load fine tuned model.
model.load_weights('bert_model.h5')
print("Predicting.")
predict_vec = np.memmap('vectors/dev_bert.mm', dtype='float32', mode='w+', shape=(25000, 1000))
predict_vec2 = np.memmap('vectors/test_bert.mm', dtype='float32', mode='w+', shape=(25000, 1000))
predict_vec3 = np.memmap('vectors/train_bert.mm', dtype='float32', mode='w+', shape=(50000, 1000))
# Generate predictions for train, dev and test sets and save to file for ensemble.
predict_vec[:] = model.predict(dev)[0][:]
del predict_vec
predict_vec2[:] = model.predict(test)[0][:]
del predict_vec2
predict_vec3[:] = model.predict(train)[0][:]
del predict_vec3
